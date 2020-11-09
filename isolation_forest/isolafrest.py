# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/11/4 1:51 下午
作者:liliuliu
@文件名:isolafrest.py
------------------------------------
"""
# python3
# author Silence Lu
# version 4.2.3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# get dataset ，
dataset = pd.read_csv("../../train_data_process/dataset1/train.csv", engine='python')
col = list(dataset.columns)
col.remove("id")
col.remove("label")
X_col = pd.DataFrame(dataset, columns=col)
X_col = X_col.values
# print(X_col)
# the code maybe useful in dataset ,not 1D array
rs = np.random.RandomState(64)
lendata = dataset.shape[0]
ifmodel = IsolationForest(n_estimators=100, verbose=2, n_jobs=2, max_samples=256, random_state=rs, max_features=19)
ifmodel.fit(X_col)
Iso_anomaly_score = ifmodel.decision_function(X_col)
Iso_predict = ifmodel.predict(X_col)
# Iso_anomaly_score 异常分数 评分越低，则越是异常
Iso_anomaly_score = np.column_stack((dataset[["id", "label"]], Iso_anomaly_score))
Iso_predict = np.column_stack((dataset[["id", "label"]], Iso_predict))
Iso_predict = Iso_predict.tolist()
Iso_anomaly_score = Iso_anomaly_score.tolist()
# print(Iso_date)
# threshold 阈值定义 也可以 根据实际情况手动添加也可以根据不同情况以公式表示，这里我设置为-0.1
threshold = -0.009
# print("隔离森林检测可能为异常数据：[id,异常评分],评分越低越可能是异常数据")
below_threshold_count = 0
pos_count = 0
all_pos_count = 0
for i in Iso_anomaly_score:
    if i[1] == 1:
        all_pos_count += 1
    if i[2] < threshold:
        below_threshold_count += 1
        if i[1] == 1:
            pos_count += 1
pre = pos_count / below_threshold_count
recall = pos_count / all_pos_count
f1 = 2 * (pre * recall) / (pre + recall)
print(below_threshold_count,all_pos_count, pos_count)
print(f1)
