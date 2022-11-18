# MusicBERT
[MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf), by Mingliang Zeng, Xu Tan, Rui Wang, Zeqian Ju, Tao Qin, Tie-Yan Liu, ACL 2021, is a large-scale pre-trained model for symbolic music understanding. It has several mechanisms including OctupleMIDI encoding and bar-level masking strategy that are specifically designed for symbolic music data, and achieves state-of-the-art accuracy on several music understanding tasks, including melody completion, accompaniment suggestion, genre classification, and style classification.

Projects using MusicBERT:

* [midiformers](https://github.com/tripathiarpan20/midiformers): a customized MIDI music remixing tool with easy interface for users. ([notebook](https://colab.research.google.com/drive/1C7jS-s1BCWLXiCQQyvIl6xmCMrqgc9fg?usp=sharing))

<!-- ![img](../img/musicbert_structure.PNG)  ![img](../img/musicbert_encoding.PNG)-->

<p align="center"><img src="../img/musicbert_structure.PNG" width="800"><br/> Model structure of MusicBERT </p>
<p align="center"><img src="../img/musicbert_encoding.PNG" width="500"><br/> OctupleMIDI encoding </p>

## 1. Preparing datasets

### 1.1 subjective feature regression dataset (XAI)
* Generate the dataset in OctupleMIDI format using the midi to genre mapping file with `gen_genre.py`.

  ```bash
  python -u gen_xai.py
  ```

* Binarize the raw text format dataset. (this script will read `topmagd_data_raw` folder and output `topmagd_data_bin`)

  ```bash
  bash binarize_genre.sh topmagd
  ```

## 2. Training


### 2.1 Pre-training

```bash
bash train_mask.sh lmd_full small
```

* **Download our pre-trained checkpoints here: [small](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_small.pt) and [base](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_base.pt), and save in the ` checkpoints` folder. (a newer version of fairseq is needed for using provided checkpoints: see [issue-37](https://github.com/microsoft/muzic/issues/37) or [issue-45](https://github.com/microsoft/muzic/issues/45))**



### 2.2 Fine-tuning on melody completion task and accompaniment suggestion task

```bash
bash train_nsp.sh next checkpoints/checkpoint_last_musicbert_base.pt
```

```bash
bash train_nsp.sh acc checkpoints/checkpoint_last_musicbert_small.pt
```

### 2.3 Fine-tuning on genre and style classification task

```bash
bash train_genre.sh topmagd 13 0 checkpoints/checkpoint_last_musicbert_base.pt
```

```bash
bash train_genre.sh masd 25 4 checkpoints/checkpoint_last_musicbert_small.pt
```

## 3. Evaluation

### 3.1 Melody completion task and accompaniment suggestion task

```bash
python -u eval_nsp.py checkpoints/checkpoint_last_nsp_next_checkpoint_last_musicbert_base.pt next_data_bin
```

### 3.2 Genre and style classification task

```bash
python -u eval_genre.py checkpoints/checkpoint_last_genre_topmagd_x_checkpoint_last_musicbert_small.pt topmagd_data_bin/x
```

# Jongho
- fairseq debug: fairseq_cli train.py