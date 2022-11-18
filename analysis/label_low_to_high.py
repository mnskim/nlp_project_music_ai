import csv
from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

"""
HYPOTHESIS 2

high deviation on value of low labels -> higher value of abstract-level labels?
for each performance,
get stdev: tempo, articulation, pedal
get value: "Imaginative", "Ethereal", "Convincing"

"""

LABEL_LIST = ["Regular beat change", "Long", "Cushioned", "Saturated (wet)", "Clean", "Subtle change", "Even", "Rich", "Bright", 
"Pure", "Soft", "Sophisticated(mellow)", "balanced", "Large range of dynamic", "Fast paced", "Flowing", "Swing(Flexible)", "Flat", "Harmonious", "Optimistic(pleasant)", "HIgh Energy", 
"Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]

selected_label_list = ["Regular beat change", "Long", "Cushioned", "Saturated (wet)" , "Clean", "Subtle change", "Imaginative", "Ethereal", "Convincing"]

LABEL_MAP = {i: label for i, label in enumerate(selected_label_list)}




file = open('/data1/jongho/muzic/musicbert/total.csv', encoding='utf-8')

csvreader = csv.reader(file)
header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

# sort by each segments
music_label_map = defaultdict(list)
for row in rows:
    file_name = row[2].split(".")[0]
    label_row = row[6:-2]
    if "" in label_row or 0 in label_row:
        continue
    for idx, elem in enumerate(label_row):
        label_row[idx] = float(elem)
    music_label_map[file_name].append(label_row)

print(f"#music segments: {len(music_label_map)}")

music_label_map_std = dict()
for key, annot_list in music_label_map.items():
    annot_list = np.array(annot_list)
    stdev = np.std(annot_list, axis = 0)
    stdev = stdev[:6] # low level
    average = np.average(annot_list, axis = 0)
    average = average[-3:]
    if 0 in stdev:
        continue
    music_label_map_std[key] = np.concatenate((stdev, average))

# correlation
#https://hleecaster.com/ml-multiple-linear-regression-example/
all_array = np.stack([v for v in music_label_map_std.values()], axis = 0)
df = pd.DataFrame(all_array)
df = df.rename(columns=LABEL_MAP)
x = df[selected_label_list[:6]]
for i in range(0,3):
    y = df[selected_label_list[6+i]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    print(mlr.coef_) 
    y_predict = mlr.predict(x_test)
    print(mlr.score(x_train, y_train))
    print(mlr.score(x_test, y_test))

all_array = np.stack([v for v in music_label_map_std.values()], axis = 0)
df = pd.DataFrame(all_array)
df = df.rename(columns=LABEL_MAP)
x = df[selected_label_list[:6]]
for i in range(0,3):
    y = df[selected_label_list[6+i]]
    #x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
    mlr = LinearRegression()
    mlr.fit(x, y)
    print(mlr.coef_) 
    #y_predict = mlr.predict(x_test)
    print(mlr.score(x, y))
    #print(mlr.score(x_test, y_test))

plt.scatter(df[["Regular beat change"]], df[["Ethereal"]], alpha=0.4)
plt.show()

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

# a = np.array(corr)
# idxs = np.argsort(a.ravel())[-48:-28]
# rows, cols = idxs//28, idxs%28
# top_k = a[rows,cols]
# to_print = [(LABEL_MAP[rows[i]], LABEL_MAP[cols[i]], top_k[i]) for i in range(0,len(rows),2)]
# print(to_print)
