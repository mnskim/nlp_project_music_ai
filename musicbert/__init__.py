# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
from sklearn.metrics import mean_squared_error, r2_score
import fairseq.tasks.sentence_prediction
import fairseq.tasks.masked_lm
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.models import FairseqEncoder
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from fairseq.data import (MaskTokensDataset,
                          LanguagePairDataset,
                          PrependTokenDataset,
                          data_utils)
from fairseq.models import register_model, register_model_architecture, BaseFairseqModel
from fairseq.models.roberta import TransformerSentenceEncoder, RobertaEncoder, RobertaModel
from musicbert.roberta.model import RobertaRegressionHead
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
from functools import lru_cache
from typing import Optional, Tuple
import numpy as np
import math
import logging
import os
import json
from fairseq import utils
import torch
from torch import Tensor
from typing import Union, Callable
from itertools import count

def exists(value):
    return value is not None

def default(value, default):
    if exists(value):
        return value
    return default

def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        input = F.logsigmoid(input), 
        target =target,
        reduction=reduction,
    )


logger = logging.getLogger(__name__)
disable_cp = 'disable_cp' in os.environ
print('disable_cp =', disable_cp)
mask_strategy = os.environ['mask_strategy'].split(
    '+') if 'mask_strategy' in os.environ else ['bar']
print('mask_strategy =', mask_strategy)
assert all(item in ['element', 'compound', 'bar'] for item in mask_strategy)
convert_encoding = os.environ['convert_encoding'] if 'convert_encoding' in os.environ else 'OCTMIDI'
print('convert_encoding =', convert_encoding)
crop_length = int(os.environ['crop_length']
                  ) if 'crop_length' in os.environ else None
print('crop_length =', crop_length)  # of compound tokens
max_bars = 256
max_instruments = 256


# Thank GitHub user @neelansh for providing multi-label classification solution
# See https://github.com/pytorch/fairseq/issues/2169
@register_task("xai")
class MusicBERTSentencePredictionMultilabelTaskXAI(SentencePredictionTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-cls-classes",
            type=int,
            default=-1,
            help="number of class targets",
        )
        parser.add_argument(
            "--num-reg-classes",
            type=int,
            default=-1,
            help="number of regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False) #don't use
        parser.add_argument("--no-shuffle", action="store_true", default=False) #don't use
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )

    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, 'input0', split)
        input0 = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        if self.args.init_token is not None:
            input0 = OctupleTokenDataset(input0)
        src_dataset = input0
        labels, label_lengths = [], []
        with open(os.path.join(self.args.data, 'label', split+".label")) as file:
            for line in file:
                line = line.strip()
                label = json.loads(line)
                label = torch.tensor(label)
                labels.append(label)
                label_lengths.append(len(label))
                #assert len(label) == self.args.num_reg_classes + 1, print(len(label), self.args.num_reg_classes)
        assert len(src_dataset) == len(labels)
        self.datasets[split] = LanguagePairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.label_dictionary,
            tgt=labels,
            tgt_sizes=torch.tensor(label_lengths),
            tgt_dict=self.label_dictionary,
            left_pad_source=False,
            input_feeding=False,
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_cls_classes > 0, "Must set --num-cls-classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, "label", "dict.txt"),
                source=False,
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(args, data_dict, label_dict)

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_cls_classes,
        )
        if self.args.num_reg_classes > 1:
            model.register_regression_head(
                getattr(args, "regression_head_name", "sentence_regression_head"),
                num_classes=self.args.num_reg_classes,
            )

        return model

@register_criterion("M2P_xai")
class MusicBERTM2PCriterionForXAI(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])
        #sample_size = targets.numel()
        sample_size = logits.size()[0]

        targets = targets[:,-1]
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(logits, targets.long())

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = logits.argmax(dim=1)
        logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

@register_criterion("M2PF_xai_adv")
class MusicBERTSentencePredictionMultilabelCriterionForXAIADV(SentencePredictionCriterion):
    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task, classification_head_name, regression_target)
        self.loss_fn = nn.MSELoss()
        self.loss_last_fn = nn.MSELoss()
        self.gold_loss_fn = nn.MSELoss()
        self.gold_loss_last_fn = nn.MSELoss()
        self.alpha = 1
        self.num_steps = 1
        self.step_size = 1e-3
        self.epsilon = 1e-6
        self.noise_var = 1e-5


    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"
        #print(sample["net_input"])
        embeds = model.encoder.sentence_encoder.embed_tokens(sample["net_input"]["src_tokens"])

        logits, _ = model(
            **sample["net_input"],
            token_embeddings = embeds,
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        targets = model.get_targets(sample, [logits])
        targets = targets[:,:-1]
        logits = torch.sigmoid(logits)

        #FIXME: virtual loss 의 last loss fn은 MSE or KL div
        virtual_loss = self.get_perturbed_loss(
            embeds, logits, model, sample, 
            loss_fn=self.loss_fn, loss_last_fn=self.loss_last_fn
        )

        labels_loss = self.get_perturbed_loss(
            embeds, targets.float(), model, sample,
            loss_fn=self.gold_loss_fn, loss_last_fn=self.gold_loss_last_fn,
        )

        loss = labels_loss + self.alpha * virtual_loss

        sample_size = logits.size()[0]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size * logits.size()[1],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds = F.relu(torch.sign(logits))
        #logging_output["ncorrect"] = sample_size - \
        #    torch.sign((preds != targets).sum(dim=1)).sum().data
        logging_output["y_true"] = targets.detach().cpu().numpy()
        logging_output["y_pred"] = logits.detach().cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.vstack(tuple(log.get("y_pred")
                                     for log in logging_outputs if "y_pred" in log))
            y_true = np.vstack(tuple(log.get("y_true")
                                     for log in logging_outputs if "y_true" in log))
            metrics.log_scalar("MSE", mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)), round=4)
            metrics.log_scalar("R2", r2_score(y_true.reshape(-1), y_pred.reshape(-1)), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

    @torch.enable_grad()
    def get_perturbed_loss(
        self, embeds: Tensor, state: Tensor, model, sample, loss_fn: Callable, loss_last_fn: Callable,
    ):
        noise = torch.randn_like(embeds, requires_grad=True) * self.noise_var
        for i in count():
            # Compute perturbed embed and states
            embed_perturbed = embeds + noise

            state_perturbed, _ = model(
                **sample["net_input"],
                token_embeddings = embed_perturbed,
                features_only=True,
                classification_head_name=self.classification_head_name,
            )

            # Return final loss if last step (undetached state)
            if i == self.num_steps:
                return loss_last_fn(state_perturbed, state)
            # Compute perturbation loss (detached state)
            loss = loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            (noise_gradient,) = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible
            step = noise + self.step_size * noise_gradient
            # Normalize new noise step into norm induced ball
            step_norm = self.inf_norm(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()
    def inf_norm(self, x):
        return torch.norm(x, p=float("inf"), dim=-1, keepdim=True)

@register_criterion("M2PF_xai")
class MusicBERTSentencePredictionMultilabelCriterionForXAI(SentencePredictionCriterion):
    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"
        #print(sample["net_input"])
        #embeds = model.encoder.sentence_encoder(sample["net_input"]["src_tokens"])
        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits])
        targets = targets[:,:-1]
        logits = torch.sigmoid(logits)
        # loss = F.binary_cross_entropy_with_logits(
        #     logits, targets.float(), reduction='sum')
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, targets.float())
        sample_size = logits.size()[0]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size * logits.size()[1],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        #logging_output["ncorrect"] = sample_size - \
        #    torch.sign((preds != targets).sum(dim=1)).sum().data
        logging_output["y_true"] = targets.detach().cpu().numpy()
        logging_output["y_pred"] = logits.detach().cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred" in logging_outputs[0]:
            y_pred = np.vstack(tuple(log.get("y_pred")
                                     for log in logging_outputs if "y_pred" in log))
            y_true = np.vstack(tuple(log.get("y_true")
                                     for log in logging_outputs if "y_true" in log))
            metrics.log_scalar("MSE", mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)), round=4)
            metrics.log_scalar("R2", r2_score(y_true.reshape(-1), y_pred.reshape(-1)), round=4)
            #metrics.log_scalar("R2", r2_score(y_true, y_pred), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False

@register_criterion("M2PFnP_xai")
class MusicBERTM2PFnPCriterionForXAI(SentencePredictionCriterion):
    def __init__(self, task, classification_head_name, regression_target, regression_head_name):
        print("****", task, classification_head_name, regression_target, regression_head_name)
        super().__init__(task, classification_head_name, regression_target)
        self.classification_head_name = classification_head_name
        self.regression_head_name = regression_head_name

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--regression-head-name',
                            default='sentence_regression_head',
                            help='name of the regression head to use')

    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=M2PFnP"
        assert (
            hasattr(model, "regression_heads")
            and self.regression_head_name in model.regression_heads
        ), "model must provide sentence classification head for --criterion=M2PFnP"
        (logits_cls, logits_reg) , _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
            regression_head_name=self.regression_head_name,
        )
        #print(model.__attr__)
        #print(model.__dict__)
        targets = model.get_targets(sample, [logits_reg])
        targets_reg = targets[:,:-1]
        targets_cls = targets[:,-1]
        logits_reg = torch.sigmoid(logits_reg)
        # loss = F.binary_cross_entropy_with_logits(
        #     logits, targets.float(), reduction='sum')
        loss_reg_fct = nn.MSELoss(reduction='sum')
        loss_cls_fct = nn.CrossEntropyLoss(reduction='sum')
        loss_reg = loss_reg_fct(logits_reg, targets_reg.float())
        loss_cls = loss_cls_fct(logits_cls, targets_cls.long()) 
        #print(f"loss_reg: {loss_reg}, loss_cls: {loss_cls}")
        loss = loss_reg + loss_cls
        sample_size = logits_reg.size()[0]
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size * logits_reg.size()[1],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        preds_cls = logits_cls.argmax(dim=1)
        logging_output["ncorrect"] = (preds_cls == targets_cls).sum()
        logging_output["y_true_cls"] = targets_cls.detach().cpu().numpy()
        logging_output["y_pred_cls"] = preds_cls.detach().cpu().numpy()
        logging_output["y_true_reg"] = targets_reg.detach().cpu().numpy()
        logging_output["y_pred_reg"] = logits_reg.detach().cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
        if len(logging_outputs) > 0 and "y_pred_reg" in logging_outputs[0]:
            y_pred = np.vstack(tuple(log.get("y_pred_reg")
                                     for log in logging_outputs if "y_pred_reg" in log))
            y_true = np.vstack(tuple(log.get("y_true_reg")
                                     for log in logging_outputs if "y_true_reg" in log))
            metrics.log_scalar("MSE", mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)), round=4)
            metrics.log_scalar("R2", r2_score(y_true.reshape(-1), y_pred.reshape(-1)), round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False


class OctupleMaskTokensDataset(MaskTokensDataset):
    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            sz = len(item)
            assert (
                self.mask_idx not in item
            ), "Dataset contains mask_idx (={}), this is not expected!".format(
                self.mask_idx,
            )
            assert not self.mask_whole_words, 'mask whole words not supported for cp'

            def generate_mask(sz, prob):
                mask_n = np.random.rand(sz)
                mask_s = np.zeros(sz, dtype=np.int8)
                mask_s += mask_n < prob * \
                    (self.random_token_prob)  # 3 -> random
                mask_s += mask_n < prob * \
                    (self.random_token_prob +
                     self.leave_unmasked_prob)  # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s
            mask_prob = self.mask_prob
            mask = np.zeros_like(item, dtype=np.int8)
            # mask bos eos tokens (compound)
            mask[:8] = np.repeat(generate_mask(1, mask_prob), 8)
            # mask bos eos tokens (compound)
            mask[-8:] = np.repeat(generate_mask(1, mask_prob), 8)
            strategy = np.random.choice(mask_strategy)
            if strategy == 'element':  # element level mask
                mask[8: -8] = np.repeat(generate_mask(sz -
                                                      2 * 8, mask_prob), 1)
            if strategy == 'compound':  # compound token level mask
                mask[8: -8] = np.repeat(generate_mask(sz //
                                                      8 - 2, mask_prob), 8)
            if strategy == 'bar':  # bar level mask
                mask[8: -8] = generate_mask((max_bars * max_instruments + len(self.vocab)) * 8, mask_prob).reshape(-1, 8)[
                    ((item[8: -8: 8] - 4) * max_instruments) + (item[8 + 2: -8 + 2: 8] - 4)].flatten()
            if self.return_masked_tokens:
                new_item = item.numpy()[:]
                new_item[mask == 0] = self.pad_idx
                return torch.from_numpy(new_item)
            masked_item = np.random.choice(len(self.vocab), sz)
            set_original = np.isin(mask, [0, 2])
            masked_item[set_original] = item[set_original]
            set_mask = np.isin(mask, [1])
            masked_item[set_mask] = self.mask_idx
            return torch.from_numpy(masked_item)


class OctupleEncoder(TransformerSentenceEncoder):
    def __init__(self, *args, **kwargs) -> None:
        self.adv_training = kwargs.pop('adv_training')
        super().__init__(*args, **kwargs)
        self.tpu = False
        embedding_dim = kwargs['embedding_dim']
        if not disable_cp:
            self.downsampling = nn.Sequential(
                nn.Linear(embedding_dim * 8, embedding_dim))
            self.upsampling = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 8))
        self.attn_mask = None
        self.num_attention_heads = kwargs['num_attention_heads']

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None, # None 
        last_state_only: bool = False, # True
        positions: Optional[torch.Tensor] = None, # None
        token_embeddings: Optional[torch.Tensor] = None, # None으로 들어옴
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ratio = 1 if disable_cp else 8
        if not disable_cp: #disable_cp=False
            assert tokens.shape[1] % ratio == 0, 'token sequences length should be multiple of ' + str(
                ratio) + ' for compound mode'
            assert last_state_only, 'hidden states not available for compound mode'
            assert positions is None, 'custom positions is not supported for compound mode'
            #assert token_embeddings is None, 'custom token embeddings is not supported for compound mode'
            assert segment_labels is None, 'segment embedding not supported for compound mode'
        padding_mask = tokens[:, ::ratio].eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None
        if token_embeddings is not None:
            x = token_embeddings
            #print('use custom token embedding')
        else:
            x = self.embed_tokens(tokens)
        if not disable_cp:
            x = self.downsampling(x.view(x.shape[0], x.shape[1] // ratio, -1))
        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.embed_positions is not None:
            x = x + \
                self.embed_positions(tokens[:, ::ratio], positions=positions)
        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        # why transpose?
        x = x.transpose(0, 1)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            #x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask = self.attn_mask)[0]
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)
        if not disable_cp:
            x = x.transpose(0, 1)
            x = self.upsampling(x).view(x.shape[0], x.shape[1] * ratio, -1)
            x = x.transpose(0, 1)
        sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]
        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            #print(len(inner_states), inner_states[0].shape)
            #print(sentence_rep.shape)
            return inner_states, sentence_rep

class MusicBERTEncoder(RobertaEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.sentence_encoder = OctupleEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            adv_training = args.adv,
        )
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        token_embeddings = None,
        **unused,
    ):
        #print("unused:", kwargs)
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens, token_embeddings = token_embeddings
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra        

    def extract_features(self, src_tokens, return_all_hiddens=False, token_embeddings=None):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
            token_embeddings=token_embeddings,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {"inner_states": inner_states if return_all_hiddens else None}


@register_model("musicbert")
class MusicBERTModel(RobertaModel):

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.regression_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--spectral-norm-regression-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the regression head",
        )
        
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample
        encoder = MusicBERTEncoder(args, task.source_dictionary)
        return cls(args, encoder)
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        regression_head_name = None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        #print("musicbertmodelforawrd", kwargs.keys())
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, token_embeddings = kwargs.get("token_embeddings", None))
        if classification_head_name is not None:
            x1 = self.classification_heads[classification_head_name](x)
            if regression_head_name is not None: #M2PFnP
                x2 = self.regression_heads[regression_head_name](x)
                return (x1, x2), extra
            else:
                return x1, extra
        else:
            return x, extra

    def register_regression_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a regression head."""
        if name in self.regression_heads:
            prev_num_classes = self.regression_heads[name].out_proj.out_features
            prev_inner_dim = self.regression_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        # can be changed to custom regression head
        self.regression_heads[name] = RobertaRegressionHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_regression_head,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        #print(state_dict.keys())
        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


        # Handle new regression heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "regression_heads")
            else self.regression_heads.keys()
        )
        #print(current_head_names)
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "regression_heads."):
                continue
            
            head_name = k[len(prefix + "regression_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "regression_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "regression_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_regression_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting regression head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.regression_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.regression_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting regression head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added regression heads into the state dict
        # with their current weights.
        if hasattr(self, "regression_heads"):
            cur_state = self.regression_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "regression_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "regression_heads." + k)
                    state_dict[prefix + "regression_heads." + k] = v

@register_model_architecture("musicbert", "musicbert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )
    args.spectral_norm_regression_head = getattr(
        args, "spectral_norm_regression_head", False
    )
    args.adv = getattr(args, "adv", False)


@register_model_architecture("musicbert", "musicbert_base")
def musicbert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_large")
def musicbert_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_medium")
def musicbert_medium_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_small")
def musicbert_small_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_mini")
def musicbert_mini_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    base_architecture(args)


@register_model_architecture("musicbert", "musicbert_tiny")
def musicbert_tiny_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    base_architecture(args)


class OctupleTokenDataset(PrependTokenDataset):
    def adaptor(self, e):
        prev_bar = None
        prev_pos = None
        prev_prog = None
        new_e = []
        for i in e:
            if prev_bar != i[0]:
                prev_bar = i[0]
                prev_pos = None
                new_e.append((i[0], None, None, None, None, None, i[6], None))
            if prev_pos != i[1]:
                prev_pos = i[1]
                prev_prog = None
                new_e.append((None, i[1], None, None, None, None, None, i[7]))
            if prev_prog != i[2]:
                prev_prog = i[2]
                new_e.append((None, None, i[2], None, None, None, None, None))
            if True:
                new_e.append((None, None, None, i[3], i[4], i[5], None, None))
        return new_e

    def convert(self, item):
        encoding = item[8: -8].tolist()
        encoding = list(tuple(encoding[i: i + 8])
                        for i in range(0, len(encoding), 8))
        encoding = self.adaptor(encoding)
        if convert_encoding == 'CP':
            encoding = list(3 if j is None else j for i in encoding for j in i)[
                :crop_length * 8]
        elif convert_encoding == 'REMI':
            encoding = list(j for i in encoding for j in i if j is not None)[
                :crop_length]
        else:
            assert False, 'Unknown encoding format'
        bos = 0
        eos = 2
        encoding = ([bos] * 8) + encoding + ([eos] * 8)
        return torch.tensor(encoding)

    def __init__(self, dataset, token=None):
        super().__init__(dataset, token=None)
        if convert_encoding != 'OCTMIDI':
            self._sizes = np.array([len(self.convert(i)) for i in dataset])
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if convert_encoding != 'OCTMIDI':
            item = self.convert(item)
        return item

    def num_tokens(self, index):
        return self._sizes[index].item()

    def size(self, index):
        return self._sizes[index].item()


fairseq.tasks.sentence_prediction.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.PrependTokenDataset = OctupleTokenDataset
fairseq.tasks.masked_lm.MaskTokensDataset = OctupleMaskTokensDataset
