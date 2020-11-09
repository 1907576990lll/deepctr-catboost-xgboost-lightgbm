# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/11/3 2:48 下午
作者:liliuliu
@文件名:catboost_multi_data.py
------------------------------------
"""
# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/10/31 10:47 上午
作者:liliuliu
@文件名:catboost.py
------------------------------------
"""
from catboost import CatBoostClassifier, Pool, cv
import numpy as np
import pandas as pd


def pd_standardize(df, numeric_columns, mean=False, std=True):
    if mean == True and std == True:
        # zscore
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    elif mean == False and std == True:
        df[numeric_columns] = df[numeric_columns] / df[numeric_columns].std()
    elif mean == True and std == False:
        df[numeric_columns] = df[numeric_columns] - df[numeric_columns].mean()
    return df


base_info_cat = ["oplocdistrict", "industryphy", "industryco", "enttype", "enttypeitem",
                 "state", "orgid", "jobid", "adbusign", "townsign", "regtype", "compform", "opform", "venind",
                 "enttypegb", "oploc"]
base_info_language = ["dom", "opscope"]
base_info_num = ["opfrom", "regcap", "empnum"]

annual_report_info_cat = ["ANCHEYEAR", "STATE", "EMPNUMSIGN", "BUSSTNAME", "WEBSITSIGN", "FORINVESTSIGN",
                          "STOCKTRANSIGN", "PUBSTATE"]
annual_report_info_num = ["FUNDAM", "EMPNUM", "COLGRANUM", "RETSOLNUM", "DISPERNUM", "COLEMPLNUM",
                          "RETEMPLNUM",
                          "DISEMPLNUM", "UNEEMPLNUM"]
change_info_cat = ["128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138",
                   "907", "902", "908",
                   "903", "143", "144", "140", "150", "921",
                   "922", "925", "930", "935", "939", "172", "301", "180", "901", "190", "110",
                   "111", "112", "113",
                   "114", "115", "116", "117", "118", "119",
                   "120", "121", "124", "125", "126"]
change_info_num = ["mean_time_span", "near", "far"]

entprise_evaluate_df = pd.read_csv("../../data/entprise_evaluate.csv")

train_df0 = pd.read_csv("../../train_data_process/dataset4/train0.csv")
test_df0 = pd.read_csv("../../train_data_process/dataset4/test0.csv")

train_df1 = pd.read_csv("../../train_data_process/dataset4/train1.csv")
test_df1 = pd.read_csv("../../train_data_process/dataset4/test1.csv")

train_df2 = pd.read_csv("../../train_data_process/dataset4/train2.csv")
test_df2 = pd.read_csv("../../train_data_process/dataset4/test2.csv")
# train_df = pd_standardize(train_df, data1_num_columns)
# test_df = pd_standardize(test_df, data1_num_columns)
all_columns0 = base_info_cat + base_info_num
cat_columns0 = base_info_cat

all_columns1 = annual_report_info_cat + annual_report_info_num
cat_columns1 = annual_report_info_cat

all_columns2 = change_info_cat + change_info_num
cat_columns2 = change_info_cat
# 类别特征只能是整型，或者字符串
train_df0[cat_columns0] = train_df0[cat_columns0].astype('int')
test_df0[cat_columns0] = test_df0[cat_columns0].astype('int')

train_df1[cat_columns1] = train_df1[cat_columns1].astype('int')
test_df1[cat_columns1] = test_df1[cat_columns1].astype('int')

train_df2[cat_columns2] = train_df2[cat_columns2].astype('int')
test_df2[cat_columns2] = test_df2[cat_columns2].astype('int')

cat_column_indices0 = [train_df0[all_columns0].columns.get_loc(c) for c in cat_columns0]
cat_column_indices1 = [train_df1[all_columns1].columns.get_loc(c) for c in cat_columns1]
cat_column_indices2 = [train_df2[all_columns2].columns.get_loc(c) for c in cat_columns2]

# 需要替换的参数cb自身参数一套。数据集参数all_columns,cat_columns,cat_column_indices,train_df,test_df,结果存储文件test.csv
base_info_parm = {"iter": 250, "depth": 3, "learning_rate": 0.1, "all_columns": all_columns0,
                  "cat_columns": cat_columns0,
                  "cat_column_indices": cat_column_indices0, "train_df": train_df0, "test_df": test_df0,
                  "test_result": "test0.csv"}
annual_info_parm = {"iter": 6000, "depth": 5, "learning_rate": 0.3, "all_columns": all_columns1,
                    "cat_columns": cat_columns1,
                    "cat_column_indices": cat_column_indices1, "train_df": train_df1, "test_df": test_df1,
                    "test_result": "test1.csv"}
change_info_parm = {"iter": 10000, "depth": 5, "learning_rate": 0.3, "all_columns": all_columns2,
                    "cat_columns": cat_columns2,
                    "cat_column_indices": cat_column_indices2, "train_df": train_df2, "test_df": test_df2,
                    "test_result": "test2.csv"}
for dic in [base_info_parm, annual_info_parm, change_info_parm]:
    deep = dic["depth"]
    lr = dic["learning_rate"]
    train_df = dic["train_df"]
    test_df = dic["test_df"]
    all_columns = dic["all_columns"]
    cat_columns = dic["cat_columns"]
    cat_column_indices = dic["cat_column_indices"]
    test_result_csv = dic["test_result"]
    iter = dic["iter"]
    cb = CatBoostClassifier(
        iterations=iter,
        task_type='CPU',
        # 'rsm':0.1,
        boosting_type='Ordered',
        objective="Logloss",
        # custom_metric= ['AUC'],
        eval_metric='F1',
        one_hot_max_size=255,
        learning_rate=lr,
        scale_pos_weight=1,
        # 'l2_leaf_reg': 1,
        depth=deep,
        max_bin=255,
        has_time=False,
        random_seed=42,
    )
    cv_data = cv(
        Pool(train_df[all_columns], train_df["label"], cat_features=cat_column_indices),
        cb.get_params(), nfold=3, verbose_eval=100
    )

    cb.fit(
        train_df[all_columns],
        train_df["label"],
        cat_features=cat_columns,
        verbose_eval=100,
        metric_period=100,
        early_stopping_rounds=1000,
    )
    train_pool = Pool(train_df[all_columns], train_df["label"], cat_features=cat_columns)
    feature_importances = cb.get_feature_importance(train_pool)
    feature_names = all_columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))
    res = cb.predict_proba(test_df[all_columns])
    scores = np.asarray([pairs[1] for pairs in res])
    test_df["score"] = scores
    print(len([i for i in scores if i > 0.5]),[i for i in scores if i > 0.5])
    uploaded = entprise_evaluate_df.merge(test_df, on='id', how='left')
    uploaded[["id", "score"]] = uploaded[["id", "score_y"]]
    print(uploaded)
    uploaded[["id", "score"]].to_csv(test_result_csv, sep=",", index=False)
