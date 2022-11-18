import csv
from collections import defaultdict, OrderedDict
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json
"""
HYPOTHESIS 3: 16 bars - lower agreement between annotators?

"""

LABEL_LIST = ["Stable beat", "Mechanical Tempo", "Intensional", "Regular beat change", "Long", "Cushioned", "Saturated (wet)", "Clean", "Subtle change", "Even", "Rich", "Bright", 
"Pure", "Soft", "Sophisticated(mellow)", "balanced", "Large range of dynamic", "Fast paced", "Flowing", "Swing(Flexible)", "Flat", "Harmonious", "Optimistic(pleasant)", "HIgh Energy", 
"Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]
LABEL_LIST = [LABEL_LIST[0]] + LABEL_LIST[4:]
print(len(LABEL_LIST))
PIANIST_MAP = OrderedDict()
def estimate_maxima(data):
    if len(set(data))<=1: # all datas are equal
        return data[0]
    kde = gaussian_kde(data)
    no_samples = 50
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    #maxima_index = probs.argmax()
    # in case if more than 1 argmaxs
    winner = np.argwhere(probs == np.amax(probs))
    maxima = np.average(samples[winner.flatten()])
    return maxima

def stdev_comparison():
    file = open('/data1/jongho/muzic/musicbert/processed/total.csv', encoding="utf-8")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)
    dontknow_8 = []
    dontknow_8_rows = []
    dontknow_16 = []
    dontknow_16_rows = []

    music_label_map = defaultdict(list)
    for row in rows:
        user = row[0]
        file_name =   row[2].split(".")[0]
        #label_row = row[3:-2]
        label_row = [row[3]] + row[7:-2] # skip 1-1 ~ 1-4
        for idx, elem in enumerate(label_row):
            if elem == "":
                exit("null data")
            else:
                label_row[idx] = float(elem) / 7.0
        # skip 0
        if 0.0 in label_row:
            print('filename: ', file_name, label_row)
            if '8bars' in file_name:
                dontknow_8.append(file_name)
                dontknow_8_rows.extend(np.where(np.array(label_row) == 0.0)[0].tolist())
            elif '16bars' in file_name:
                dontknow_16.append(file_name)
                dontknow_16_rows.extend(np.where(np.array(label_row) == 0.0)[0].tolist())
            continue
        else:
            music_label_map[file_name].append(label_row)
            
    print()
    print("16bars", len(dontknow_16), "8bars", len(dontknow_8)) # 16bars 44 / 1132 8bars 330 / 6081 -> 16 bars라고 해서 don't know라 마킹하는 사람들이 더 늘어나는 건 아님
    print()
    # don't know가 어떤 피처들에 분포되어 있는지 확인하기
    

    dontknow_16_rows = dict(Counter(dontknow_16_rows))
    dontknow_16_rows = {LABEL_LIST[idx]: dontknow_16_rows[idx]/sum(dontknow_16_rows.values()) if dontknow_16_rows.get(idx) != None else 0 for idx in range(len(LABEL_LIST))}
    #dontknow_16_rows = {LABEL_LIST[idx]: dontknow_16_rows[idx]/sum(dontknow_16_rows.values()) for k in sorted(dontknow_16_rows)}
    dontknow_8_rows = dict(Counter(dontknow_8_rows))
    dontknow_8_rows = {LABEL_LIST[k]: dontknow_8_rows[k]/sum(dontknow_8_rows.values()) if dontknow_8_rows.get(idx) != None else 0 for k in sorted(dontknow_8_rows)}
    #print(dontknow_16_rows)
    #print(dontknow_8_rows)
    # dontknow_compare
    # https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
    X = list(range(len(dontknow_16_rows.keys())))
    Y = list(dontknow_8_rows.values())
    Z = list(dontknow_16_rows.values())   
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = '8bars')
    plt.bar(X_axis + 0.15, Z, 0.3, label = '16bars')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("ratio")
    plt.title("'don't know' distribution")
    plt.legend()
    plt.show()

    X = list(range(len(LABEL_LIST)))
    Y = [Y[i] - Z[i] for i in range(len(Z)) ]
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = 'difference')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("difference")
    plt.title("dontknow difference")
    plt.legend()
    plt.show()    

    music_label_map_std = dict()

    # get stdev
    for key, annot_list in tqdm(music_label_map.items()):
        annot_list = np.array(annot_list)
        if len(annot_list) == 1:
            continue
        stdevs = np.std(annot_list, axis=0)
        
        stdevs = stdevs.tolist()
        music_label_map_std[key] = stdevs
    # split to 8 bars vs 16 bars
    stdev_8 = dict()
    stdev_16 = dict()
    for k, v in music_label_map_std.items():
        if "8bars" in k:
            stdev_8[k] = v
        elif "16bars" in k:
            stdev_16[k] = v
    #stdev_8 = np.average(np.array(np.array(list(stdev_8.values()))))
    stdev_8 = np.average([np.array(v) for v in stdev_8.values()], axis=0)
    stdev_16 = np.average([np.array(v) for v in stdev_16.values()], axis=0)

    # 16 bars stdev higher?
    X = list(range(len(LABEL_LIST)))
    Y = list(stdev_8)
    Z = list(stdev_16)   
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = '8bars')
    plt.bar(X_axis + 0.15, Z, 0.3, label = '16bars')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("stdev")
    plt.title("stdev distribution")
    plt.legend()
    plt.show()

    X = list(range(len(LABEL_LIST)))
    Y = [stdev_8[i] - stdev_16[i] for i in range(len(stdev_16)) ]
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = 'difference')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("difference")
    plt.title("stdev difference")
    plt.legend()
    plt.show()    

def filter_total_csv():
    def get_bars(file_name):
        return file_name.split('/')[-1].split('.')[0].split("_")[-3]
    def get_segment_id(file_name):
        return int(file_name.split('/')[-1].split('.')[0].split("_")[-1])
    
    file = open('/data1/jongho/muzic/musicbert/processed/total.csv', encoding="utf-8")
    csvreader = csv.reader(file)
    head = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    
    music_label_map = defaultdict(list)
    for row in rows:
        user = row[0]
        file_name =   row[2].split(".")[0]
        #label_row = row[3:-2]
        label_row = [row[3]] + row[7:-2] # skip 1-1 ~ 1-4
        for idx, elem in enumerate(label_row):
            if elem == "":
                exit("null data")
            else:
                label_row[idx] = float(elem)
        # skip 0
        if 0.0 in label_row:
            continue
        else:
            music_label_map[file_name].append(label_row)

    music_label_map_filtered = dict()
    stdev_diff_dict = dict()
    avg_diff_dict = dict()
    for file_name in music_label_map:
        if get_bars(file_name) == "16bars":
            bars8_segid = str(get_segment_id(file_name) * 2 - 1) 
            if len(bars8_segid) == 1: 
                bars8_segid = "0" + bars8_segid
            bars8_name = "_".join(file_name.split("_")[:3]+ ["8bars", file_name.split("_")[4], bars8_segid])
            if bars8_name in music_label_map:
                music_label_map_filtered[bars8_name] = music_label_map[bars8_name]
                music_label_map_filtered[file_name] = music_label_map[file_name]
                
                annot_list_16 = np.array(music_label_map[file_name])
                annot_list_8 = np.array(music_label_map[bars8_name])

                avg_16 = np.average(annot_list_16, axis=0)
                avg_8 = np.average(annot_list_8, axis=0)
                avg_diff = avg_8 - avg_16
                avg_diff = avg_diff.tolist()
                avg_diff_dict[file_name] = avg_diff
                # 16 bars stdev
                if len(annot_list_16) == 1:
                    continue
                stdevs_16 = np.std(annot_list_16, axis=0)
                # 8 bars stdev
                if len(annot_list_8) == 1:
                    continue
                stdevs_8 = np.std(annot_list_8, axis=0)
                stdevs_diff = stdevs_8 - stdevs_16
                stdevs_diff = stdevs_diff.tolist()
                stdev_diff_dict[file_name] = stdevs_diff
                # check if more 0 labels in 8bar low features
                # check if low stdevs in 8bar low features
            else:
                print(file_name)

    X = list(range(len(LABEL_LIST)))
    Y = np.average([np.array(v) for v in stdev_diff_dict.values()], axis=0)
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = 'difference')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("difference")
    plt.title("difference of stdev (8bars - 16bars)")
    plt.legend()
    plt.show()    

    X = list(range(len(LABEL_LIST)))
    Y = np.average([np.array(v) for v in avg_diff_dict.values()], axis=0)
    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.15, Y, 0.3, label = 'difference')
    plt.xticks(X_axis, X)
    plt.xlabel("features")
    plt.ylabel("difference")
    plt.title("difference of mean (8bars - 16bars)")
    plt.legend()
    plt.show()

    music_label_map_apex = dict()

    # kernel density estimation
    for key, annot_list in tqdm(music_label_map_filtered.items()):
        annot_list = np.array(annot_list).transpose()
        maxima = np.array([estimate_maxima(row)/7 for row in annot_list])
        maxima = maxima.transpose().tolist()
        music_label_map_apex[key] = maxima

    # add pianist info
    for key, annot_list in tqdm(music_label_map_apex.items()):
        if key.split("_")[-2] not in PIANIST_MAP:
            PIANIST_MAP[key.split("_")[-2]] = len(PIANIST_MAP)
    print(PIANIST_MAP)
    
    for key, annot_list in tqdm(music_label_map_apex.items()):
        music_label_map_apex[key].append(PIANIST_MAP[key.split("_")[-2]])
    print(len(music_label_map_apex))
    json.dump(music_label_map_apex, open("/data1/jongho/muzic/musicbert/processed/midi_label_map_apex_8_16.json", 'w'))

if __name__ == "__main__": 
    filter_total_csv() 

