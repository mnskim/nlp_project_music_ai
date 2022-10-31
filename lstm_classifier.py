import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np
import random
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.tensorboard import SummaryWriter


# class definition
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=24, num_layers=2, dropout=0.1):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim=self.embedding_dim)
        # setup LSTM layer
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # setup output layer
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, output_dim)
        self.act = nn.GELU()
        self.loss_fct = nn.MSELoss()

    def forward(self, x, lengths, label, hidden=None):
        x = self.embedding(x)
        #print(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _outputs, (hidden, _cell) = self.lstm(x)
        logits = self.linear1(hidden[-1])              # equivalent to return_sequences=False from Keras
        logits = self.act(logits)
        logits = self.linear2(logits)
        loss = self.loss_fct(logits, label)
        loss = torch.sigmoid(loss)
        return logits, loss

class MusicDataset(Dataset):

    def __init__(self, input_file, label_file, vocabs):
        input_file = input_file.readlines()
        self.inputs = [] 
        for line in input_file:
            line = line.strip()
            tokens = line.split(" ")
            token_ids = [vocabs[token] for token in tokens]
            self.inputs.append(token_ids)
        label_file = label_file.readlines()
        self.labels = [json.loads(line.strip()) for line in label_file]
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        sample = {}
        sample['input'] = self.inputs[idx]
        sample['label'] = self.labels[idx]
        return sample

def music_collate_fn(batch):
    sample = {}
    length = [len(b['input']) for b in batch]
    perform = torch.nn.utils.rnn.pad_sequence([torch.tensor(b['input']) for b in batch], batch_first=True, padding_value=0)
    label = torch.tensor([(b['label']) for b in batch], dtype=torch.float32)
    sample['length'] = length
    sample['perform'] = perform
    sample['label'] = label
    return sample

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs",
                        default=100,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--hidden_dim",
                        default=128,
                        type=int,
                        help="hid dimension for MLP, lstm layer.")
    # parser.add_argument("--embedding_dim",
    #                     default=128,
    #                     type=int,
    #                     help="embedding")
    parser.add_argument("--num_layers",
                        default=3,
                        type=int,
                        help="hid dimension for MLP, lstm layer.")
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout",
                        default=0.2,
                        type=float,
                        help="dropout ratio")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--n_classes",
                        default=24,
                        type=int,
                        help="label class")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model",
                        type=str,
                        help="cosmos_model.bin, te_model.bin",
                        default="")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()    
    print(args)
    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:        
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print(f"device {device}, n_gpu {n_gpu}")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs                      
        torch.distributed.init_process_group(backend='nccl')

    bos_token="<s>"
    eos_token="</s>"
    vocabs = open("/data1/jongho/muzic/musicbert/xai_data_raw_apex/0/dict.txt").readlines()    
    vocabs = [voca.split(" ")[0].strip() for voca in vocabs]
    vocab_dict = {voca: i+2 for i, voca in enumerate(vocabs)}
    vocab_dict[bos_token] = 0
    vocab_dict[eos_token] = 1

    train_dataset = MusicDataset(open("/data1/jongho/muzic/musicbert/xai_data_raw_apex/0/train.txt"),
    open("/data1/jongho/muzic/musicbert/xai_data_raw_apex/0/train.label"), vocabs=vocab_dict)
    test_dataset = MusicDataset(open("/data1/jongho/muzic/musicbert/xai_data_raw_apex/0/test.txt"),
    open("/data1/jongho/muzic/musicbert/xai_data_raw_apex/0/test.label"), vocabs=vocab_dict)
    

    model = LSTM(vocab_size=len(vocab_dict), embedding_dim=args.hidden_dim,
                 hidden_dim=args.hidden_dim, output_dim=args.n_classes,
                 num_layers=args.num_layers, dropout=args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)

    train_dataloader = DataLoader(train_dataset, collate_fn=music_collate_fn, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=music_collate_fn, shuffle=False, batch_size=args.batch_size)

    #if n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")

    model.train()

    best_eval_mse = 0.0
    best_eval_r2 = -100
    global_step = 0
    if args.do_train:
        for epoch in trange(int(args.n_epochs), desc="Epoch"):
            tr_loss, tr_r2 = 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                inputs = batch['perform'].to(device)
                labels = batch['label'].to(device)
                #batch['length'] = batch['length'].to(device)
                logits, loss = model(x = inputs, lengths = batch['length'], label = labels)
                tr_loss += loss.item()
                nb_tr_steps += 1                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #nb_tr_examples += batch['label'].size(0)
                global_step += 1
                logits = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                tr_r2 += r2_score(labels.reshape(-1), logits.reshape(-1))
            
            
            print("train", tr_loss / nb_tr_steps, tr_r2 / nb_tr_steps)
            writer.add_scalar('training loss',
                            tr_loss / nb_tr_steps,
                            epoch * len(train_dataloader) + step)
            writer.add_scalar('training r2',
                            tr_r2 / nb_tr_steps,
                            epoch * len(train_dataloader) + step)

            eval_loss, nb_eval_examples, nb_eval_steps = 0.0, 0, 0
            all_preds, all_golds = [], []
            model.eval()
            for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
                inputs = batch['perform'].to(device)
                labels = batch['label'].to(device)
                #batch['length'] = batch['length'].to(device)
                with torch.no_grad():
                    eval_logits, tmp_eval_loss = model(x = inputs, lengths = batch['length'], label = labels)
                logits = eval_logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                nb_eval_examples += labels.shape[0]
                nb_eval_steps += 1    
                #eval_r2 += r2_score(logits, labels)
                eval_loss += tmp_eval_loss.item()

                all_golds.extend(labels.reshape(-1).tolist())
                all_preds.extend(logits.reshape(-1).tolist())
                
            final_r2 = r2_score(all_golds, all_preds)
            print("test", eval_loss/nb_eval_steps, final_r2)
            
            writer.add_scalar('test loss',
                eval_loss/nb_eval_steps,
                epoch * len(test_dataloader))
            writer.add_scalar('test r2',
                            final_r2, epoch * len(test_dataloader))
            # if final_r2 > best_eval_r2:
            #     print("Save at Epoch %s" % epoch)
            #     best_eval_r2 = eval_r2
            #     torch.save(model_to_save.state_dict(), output_model_file)
            
            model.train()

    if args.do_eval:
        pass
