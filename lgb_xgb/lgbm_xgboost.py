# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/11/4 5:47 下午
作者:liliuliu
@文件名:lgbm_xgboost.py
------------------------------------
"""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# XGboost
train_df = pd.read_csv("../../train_data_process/dataset1/train.csv")
submit_test_df = pd.read_csv("../../train_data_process/dataset1/test.csv")
columns = list(train_df.columns)
columns.remove("id")
columns.remove("dom")
columns.remove("label")
columns.remove("opscope")
X_train = train_df[columns]
y_train = train_df["label"]
X_test = train_df[columns][14000:]
y_test = train_df["label"][14000:]
submit_test = submit_test_df[columns]
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_test)

submit_prob_xgb = xgb.predict_proba(submit_test)
print(submit_prob_xgb[:, 1])
test_result = pd.DataFrame(columns=["id", "score"], data=None)
test_result["id"] = submit_test_df["id"].values
test_result["score"] = submit_prob_xgb[:, 1]
test_result.to_csv("test_xgb.csv")

print(classification_report(y_test, y_pred_xg))
# LGBM
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)
parameters = {'num_leaves': 2 ** 8,
              'learning_rate': 0.1,
              'is_unbalance': True,
              'min_split_gain': 0.1,
              'min_child_weight': 1,
              'reg_lambda': 1,
              'subsample': 1,
              'objective': 'binary',
              # 'device': 'gpu', # comment this line if you are not using GPU
              'task': 'train'
              }
num_rounds = 300
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test)
clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)
submit_prob_lgb = clf.predict(submit_test)
y_prob = clf.predict(X_test)
test_result = pd.DataFrame(columns=["id", "score"], data=None)
test_result["id"] = submit_test_df["id"].values
test_result["score"] = submit_prob_lgb
test_result.to_csv("test_lgb.csv")
print(y_prob)
y_pred = sklearn.preprocessing.binarize(np.reshape(y_prob, (-1, 1)), threshold=0.5)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
# AUC
xg_fpr, xg_tpr, xg_threshold = roc_curve(y_test, y_pred_xg)
lgb_fpr, lgb_tpr, lgb_threshold = roc_curve(y_test, y_pred)
plt.figure(figsize=(7, 5))
plt.title("Roc Curve")
plt.plot(xg_fpr, xg_tpr, label='XGBoost Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_xg)))
plt.plot(lgb_fpr, lgb_tpr,
         label='Light Gradient Boosting Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred)))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend()
plt.show()
