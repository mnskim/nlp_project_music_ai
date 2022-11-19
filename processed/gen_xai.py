# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from cgi import test
import os
import sys
import random
import zipfile
from multiprocessing import Pool, Manager
import preprocess as preprocess
import json
import numpy as np
import csv

#subset = input('subset: ')
JSON_PATH = 'midi_label_map_apex_reg_cls.json'
SUFFIX = '_data_raw_apex_reg_cls'
#JSON_PATH = 'midi_label_map_apex_reg_cls_rm_highstd.json'
#SUFFIX = '_data_raw_apex_reg_cls_rm_highstd'
#JSON_PATH = 'midi_label_map_apex_8_16.json'
#SUFFIX = '_data_raw_apex_8'
subset = 'xai'
raw_data_dir = subset + SUFFIX
if os.path.exists(raw_data_dir):
    if os.listdir(raw_data_dir):
        print('Output path {} already exists!'.format(raw_data_dir))
        sys.exit(0)
#data_path = input('xai dataset zip path: ')
data_path = "segmented_midi.zip"
n_folds = 5
n_times = 1  # sample train set multiple times
#max_length = int(input('sequence length: '))
max_length = 1000
preprocess.sample_len_max = max_length
preprocess.deduplicate = False
preprocess.data_zip = zipfile.ZipFile(data_path)
manager = Manager()
all_data = manager.list()
pool_num = 24

random.seed(7)

labels = dict()
with open(JSON_PATH) as f:
    for s,v in json.load(f).items():
        labels[s] = v

def get_id(file_name):
    return file_name.split('/')[-1].split('.')[0]

def get_performer_id(file_name):
    return file_name.split('/')[-1].split('.')[0].split("_")[-2]

def get_music_id(file_name):
    return file_name.split('/')[-1].split('.')[0].split("bars")[0] 

def get_bars(file_name):
    return file_name.split('/')[-1].split('.')[0].split("_")[-3]

def get_segment_id(file_name):
    return int(file_name.split('/')[-1].split('.')[0].split("_")[-1])

def get_sample(output_str_list):
    max_len = max(len(s.split()) for s in output_str_list)
    return random.choice([s for s in output_str_list if len(s.split()) == max_len])


def new_writer(file_name, output_str_list):
    if len(output_str_list) > 0:
        all_data.append((file_name, tuple(get_sample(output_str_list)
                                          for _ in range(n_times))))


preprocess.writer = new_writer


os.system('mkdir -p {}'.format(raw_data_dir))
file_list = [file_name for file_name in preprocess.data_zip.namelist(
) if file_name[-4:].lower() == '.mid' or file_name[-5:].lower() == '.midi']
file_list = [file_name for file_name in file_list if get_id(
    file_name) in labels]
random.shuffle(file_list)
label_list = [str(labels[get_id(file_name)]) for file_name in file_list]

with Pool(pool_num) as p:
    list(p.imap_unordered(preprocess.G, file_list))
random.shuffle(all_data)
#print(file_list)
#print(all_data)
print('{}/{} ({:.2f}%)'.format(len(all_data),
                               len(file_list), len(all_data) / len(file_list) * 100))

fold = 0
os.system('mkdir -p {}/{}'.format(raw_data_dir, fold))
preprocess.gen_dictionary('{}/{}/dict.txt'.format(raw_data_dir, fold))
"""split by player"""
# for cur_split in ['train', 'test']:
#     output_path_prefix = '{}/{}/{}'.format(raw_data_dir, fold, cur_split)
#     with open(output_path_prefix + '.txt', 'w') as f_txt:
#         with open(output_path_prefix + '.label', 'w') as f_label:
#             with open(output_path_prefix + '.id', 'w') as f_id:
#                 count = 0
#                 for file_name, output_str_list in all_data:
#                     if cur_split == 'test' and get_performer_id(file_name) == "12":
#                         f_txt.write(output_str_list[0] + '\n')
#                         f_label.write(
#                             json.dumps(labels[get_id(file_name)]) + '\n')
#                         f_id.write(get_id(file_name) + '\n')
#                         count += 1
#                     elif cur_split == 'train' and get_performer_id(file_name) != "12" :
#                         f_txt.write(output_str_list[0] + '\n')
#                         f_label.write(
#                             json.dumps(labels[get_id(file_name)]) + '\n')
#                         f_id.write(get_id(file_name) + '\n')
#                         count += 1
#                print(fold, cur_split, count)

"""split random"""
SPLIT_RATIO = 0.9
train_data = all_data[:int(len(all_data)*SPLIT_RATIO)]
test_data = all_data[int(len(all_data)*SPLIT_RATIO):]
for cur_split in ['train', 'test']:
    output_path_prefix = '{}/{}/{}'.format(raw_data_dir, fold, cur_split)
    with open(output_path_prefix + '.txt', 'w') as f_txt:
        with open(output_path_prefix + '.label', 'w') as f_label:
            with open(output_path_prefix + '.id', 'w') as f_id:
                count = 0
                if cur_split == "train":
                    for file_name, output_str_list in train_data:
                        f_txt.write(output_str_list[0] + '\n')
                        f_label.write(
                            json.dumps(labels[get_id(file_name)]) + '\n')
                        f_id.write(get_id(file_name) + '\n')
                        count += 1
                elif cur_split=="test": 
                    for file_name, output_str_list in test_data:
                        f_txt.write(output_str_list[0] + '\n')
                        f_label.write(
                            json.dumps(labels[get_id(file_name)]) + '\n')
                        f_id.write(get_id(file_name) + '\n')
                        count += 1
                print(fold, cur_split, count)

