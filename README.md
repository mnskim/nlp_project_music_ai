# NLP Music Project
This is the starter kit for the Music XAI Project.

A big thank you to [Jongho Kim](https://github.com/ihatedebug/) for providing the codebase!

# MusicBERT
[MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training](https://arxiv.org/pdf/2106.05630.pdf), by Mingliang Zeng, Xu Tan, Rui Wang, Zeqian Ju, Tao Qin, Tie-Yan Liu, ACL 2021, is a large-scale pre-trained model for symbolic music understanding. It has several mechanisms including OctupleMIDI encoding and bar-level masking strategy that are specifically designed for symbolic music data, and achieves state-of-the-art accuracy on several music understanding tasks, including melody completion, accompaniment suggestion, genre classification, and style classification.

Projects using MusicBERT:

* [midiformers](https://github.com/tripathiarpan20/midiformers): a customized MIDI music remixing tool with easy interface for users. ([notebook](https://colab.research.google.com/drive/1C7jS-s1BCWLXiCQQyvIl6xmCMrqgc9fg?usp=sharing))

<!-- ![img](../img/musicbert_structure.PNG)  ![img](../img/musicbert_encoding.PNG)-->

<p align="center"><img src="../img/musicbert_structure.PNG" width="800"><br/> Model structure of MusicBERT </p>
<p align="center"><img src="../img/musicbert_encoding.PNG" width="500"><br/> OctupleMIDI encoding </p>

## 0. Google drive link
- https://drive.google.com/drive/folders/1Rzncw8syf__TE5Fb1415P9V5zOcztQ5o

## 1. Preparing datasets

### 1.1 Pre-processing datasets

- please use the provided segmented midi file   `total.csv` `segment_midi.zip` since there is file name error in original Google Drive file. 
- other data `ex) metadata of annotators, original files, ... ` are in the drive
- process `total.csv` file to json file.

    ```bash
    python map_midi_to_label.py
    ```
    - File `midi_label_map_apex_reg_cls.json` is generated.
    - Currently, peak value from kernel density estimation is used as label.
    - You can also try: use all data / mean / median ... etc

- Generate XAI for music dataset in OctupleMIDI format using the midi to label mapping file with `gen_xai.py`.

    ```bash
    python -u gen_xai.py
    ```
    - Currently, train / test set is splitted by segments

- Binarize the raw text format dataset. (this script will read `xai_data_raw_apex_reg_cls` folder and output `xai_data_bin_apex_reg_cls`)

    ```bash
    bash scripts/binarize_xai.sh xai
    ```


## 2. Training

* **Download our pre-trained checkpoints here: [small](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_small.pt) and [base](https://msramllasc.blob.core.windows.net/modelrelease/checkpoint_last_musicbert_base.pt), and save in the ` checkpoints` folder. (a newer version of fairseq is needed for using provided checkpoints: see [issue-37](https://github.com/microsoft/muzic/issues/37) or [issue-45](https://github.com/microsoft/muzic/issues/45))**


### 2.1 Fine-tuning on XAI music regression task

- you should modify hyperparameters, checkpoint path, etc in sh file.

- using pre-trained model

    ```bash
    bash scripts/train_xai_base_small.sh # checkpoints/checkpoint_last_musicbert_base.pt, checkpoints/checkpoint_last_musicbert_base.pt
    ```

- If file path error, try 
``` export PYTHONPATH=`pwd` ``` 

- To custom the model, check `musicbert/__init__.py`
    - Some custom arguments are provided
    - Check [fairseq](https://fairseq.readthedocs.io/en/latest/) for more information.


- Sample script for Regression task using LSTM

    ```bash
    bash scripts/train_lstm.sh
    ```
