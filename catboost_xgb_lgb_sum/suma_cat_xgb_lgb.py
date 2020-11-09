# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/11/6 11:32 上午
作者:liliuliu
@文件名:suma_cat_xgb_lgb.py
------------------------------------
"""
import pandas as pd
import numpy as np

cat_boost_result = pd.read_csv("../catboost_mine/test1.csv")
xgb_result = pd.read_csv("../lgb_xgb/test_xgb.csv")
lgb_result = pd.read_csv("../lgb_xgb/test_lgb.csv")
cat = cat_boost_result["score"].values
cat_vote = cat
cat_bias = cat
xgb = xgb_result["score"].values
lgb = lgb_result["score"].values
for i in range(len(cat_vote)):
    np_three = np.asarray([cat[i], xgb[i], lgb[i]])
    count = np.sum(np_three > 0.5)
    # 投票
    if count == 1:
        cat_vote[i] = 0.3
    if count == 2:
        cat_vote[i] = 0.6
for i in range(len(cat_bias)):
    # 阈值
    np_three = np.asarray([cat[i], xgb[i], lgb[i]])
    ab_list = np.absolute(0.5 - np_three).tolist()
    ind = ab_list.index(max(ab_list))
    cat_bias[i] = np_three[ind]
cat_bias_df = cat_boost_result.copy()
cat_vote_df = cat_boost_result.copy()

cat_vote_df["score"] = cat_vote
cat_bias_df["score"] = cat_bias
cat_bias_df.to_csv("cat_bias.csv")
cat_vote_df.to_csv("cat_vote.csv")
