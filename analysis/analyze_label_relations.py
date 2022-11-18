import csv
from collections import defaultdict, OrderedDict
import numpy as np
#import pandas as pd
#import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt

"""HYPOTHESIS1: is 'Convincing' label can be representative of 'good play'? """

LABEL_LIST = ["Stable beat", "Mechanical Tempo", "Intensional", "Regular beat change", "Long", "Cushioned", "Saturated (wet)", "Clean", "Subtle change", "Even", "Rich", "Bright", 
"Pure", "Soft", "Sophisticated(mellow)", "balanced", "Large range of dynamic", "Fast paced", "Flowing", "Swing(Flexible)", "Flat", "Harmonious", "Optimistic(pleasant)", "HIgh Energy", 
"Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]
LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST)}


file = open('total.csv')

csvreader = csv.reader(file)
header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

# sort by each segments
music_label_map = defaultdict(list)
performer_label_map = defaultdict(list)
for row in rows:
    file_name = row[2]
    #file_name = row[2].split("_")
    #file_name = "_".join(file_name[:-2] + file_name[-1:])
    label_row = row[3:-2]
    for idx, elem in enumerate(label_row):
        if elem == "":
            label_row[idx] = 0.0
        else:
            label_row[idx] = float(elem)
    performer_id = file_name.split("_")[-2] # performer id
    music_label_map[file_name].append(label_row)
    performer_label_map[performer_id].append(label_row)

for k, v in performer_label_map.items():
    v=np.array(v)
    v=v[:,-1]
    performer_label_map[k] = np.average(v)



# Who is 'score'? 
# defaultdict(list,
#             {'1': 3.7115987460815045,
#              '2': 3.677685950413223,
#              '4': 3.8659003831417627,
#              '5': 3.49721706864564,
#              'score': 4.514056224899599,
#              '0': 3.987075928917609,
#              '3': 4.337016574585635,
#              '9': 3.5170454545454546,
#              '10': 3.759920634920635,
#              '7': 3.936823104693141,
#              '8': 4.018083182640145,
#              '6': 3.9523809523809526,
#              'Score': 4.7678100263852246,
#              '12': 3.74818401937046})

print(f"#music segments: {len(music_label_map)}")

# stdev
# TODO: is stdev a good metric?

header = header[3:-2]
stdev_list = []
for key, annot_list in music_label_map.items():
    annot_list = np.array(annot_list)
    stdev = np.std(annot_list, axis = 0) #FIXME: ignore 0 = don't know?
    stdev_list.append(stdev)
stdev_list = np.stack(stdev_list, axis=0)
stdev_avg = np.average(stdev_list, axis=0)
#stdevs = sorted({LABEL_LIST[i] : stdev_avg[i] for i in range(len(header))}.items(), key=lambda item: item[1])
stdevs = OrderedDict({LABEL_LIST[i] : stdev_avg[i] for i in range(len(header))})
#print(header)
print(stdevs)
plt.figure(figsize=(35,10))
plt.bar(stdevs.keys(), stdevs.values(),bottom=1 , color='g', label = "stdev")
plt.show()

# correlation
all_array = np.concatenate([v for v in music_label_map.values()], axis = 0)
df = pd.DataFrame(all_array)
df = df.rename(columns=LABEL_MAP)
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

a = np.array(corr)
idxs = np.argsort(a.ravel())[-48:-28]
rows, cols = idxs//28, idxs%28
top_k = a[rows,cols]
to_print = [(LABEL_MAP[rows[i]], LABEL_MAP[cols[i]], top_k[i]) for i in range(0,len(rows),2)]
print(to_print)

# multiple regression

# X = all_array[:,:-1].T
# Y = all_array[:,-1].T

# def reg_m(y, x):
#     x = np.array(x).T
#     x = sm.add_constant(x)
#     results = sm.OLS(endog=y, exog=x).fit()
#     return results
# results = reg_m(Y,X)
# print(results.summary())
# P>0.05 인 애들을 뺸다
# coef 큰애들만 찾는다

# R^2 는 설명력. 통계적 설명량이 나옴. 설명량을 늘리는 변수가 있음.
# 모델이 유의미하지 않을 수도 있음

exit()

from dominance_analysis import Dominance
dominance_regression=Dominance(data=df,target='Convincing',objective=1)
incr_variable_rsquare=dominance_regression.incremental_rsquare()
dominance_regression.plot_incremental_rsquare()
