# -*- coding = utf-8 -*-
"""
------------------------------------
@创建时间:2020/10/31 10:47 上午
作者:liliuliu
@文件名:catboost.py
------------------------------------
"""
import hyperopt
from catboost import CatBoostClassifier, Pool, cv
from numpy.random import RandomState
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
                 "enttypegb"]
base_info_language = ["dom", "opscope"]
base_info_num = ["opfrom", "regcap", "empnum"]

annual_report_info_cat = ["ANCHEYEAR", "STATE", "EMPNUMSIGN", "BUSSTNAME", "WEBSITSIGN", "FORINVESTSIGN",
                          "STOCKTRANSIGN", "PUBSTATE"]
annual_report_info_num = ["FUNDAM", "EMPNUM", "COLGRANUM", "RETSOLNUM", "DISPERNUM", "COLEMPLNUM",
                          "RETEMPLNUM",
                          "DISEMPLNUM", "UNEEMPLNUM"]
change_info_cat = ["128.0", "129.0", "130.0", "131.0", "132.0", "134.0", "135.0", "136.0", "137.0", "138.0",
                   "907.0", "902.0", "908.0",
                   "143.0", "150.0", "921.0",
                   "922.0", "925.0", "930.0", "939.0", "172.0", "301.0", "190.0", "110.0",
                   "111.0", "112.0", "113.0",
                   "114.0", "115.0", "116.0", "117.0", "118.0", "119.0",
                   "120.0", "121.0", ]
# 去掉贡献为0的列935，903，901，180，144，140，130，126，124，125
change_info_num = ["mean_time_span", "near", "far"]

data1_cat_columns = base_info_cat
data1_num_columns = base_info_num
data2_cat_columns = base_info_cat + annual_report_info_cat
data2_num_columns = base_info_num + annual_report_info_num
data3_cat_columns = base_info_cat + change_info_cat
data3_num_columns = base_info_num + change_info_num
entprise_evaluate_df = pd.read_csv("../../data/entprise_evaluate.csv")

train_df = pd.read_csv("../../train_data_process/dataset2/train.csv")
test_df = pd.read_csv("../../train_data_process/dataset2/test.csv")
# train_df = pd_standardize(train_df, data1_num_columns)
# test_df = pd_standardize(test_df, data1_num_columns)
all_columns = data2_cat_columns + data2_num_columns
cat_columns = data2_cat_columns
for i in cat_columns:  # 只能是整型，或者字符串
    train_df[i] = train_df[i].astype('int')
    test_df[i] = test_df[i].astype('int')
cat_column_indices = [train_df[all_columns].columns.get_loc(c) for c in cat_columns]


def get_best_parm():
    params_space = {
        'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 5, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
        'depth': hyperopt.hp.randint('depth', 6, 10)
    }

    trials = hyperopt.Trials()

    best = hyperopt.fmin(
        hyperopt_objective,
        space=params_space,
        algo=hyperopt.atpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=RandomState(123)
    )
    print(best)


def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=params['l2_leaf_reg'],
        learning_rate=params['learning_rate'],
        task_type='CPU',
        boosting_type='Ordered',
        objective="Logloss",
        iterations=500,
        eval_metric='AUC',
        scale_pos_weight=14,
        one_hot_max_size=255,
        loss_function='Logloss',
        random_seed=42,
        depth=params['depth'],
        max_bin=255,
        logging_level='Silent'
    )

    cv_data = cv(
        Pool(train_df[all_columns], train_df["label"], cat_features=cat_column_indices),
        model.get_params()
    )
    best_auc = np.max(cv_data['test-AUC-mean'])

    return 1 - best_auc


# get_best_parm()


cb = CatBoostClassifier(
    iterations=500,
    task_type='CPU',
    # 'rsm':0.1,
    boosting_type='Ordered',
    objective="Logloss",
    # custom_metric= ['AUC'],
    eval_metric='F1',
    one_hot_max_size=255,
    learning_rate=0.03,
    scale_pos_weight=1,
    # 'l2_leaf_reg': 1,
    depth=8,
    max_bin=255,
    has_time=False,
    random_seed=42,
)
cv_data = cv(
    Pool(train_df[all_columns], train_df["label"], cat_features=cat_column_indices),
    cb.get_params(), nfold=3, verbose_eval=100
)

# print('Best validation AUC score: {:.3f}±{:.3f} on step {}'.format(
#     np.max(cv_data['test-AUC-mean']),
#     cv_data['test-AUC-std'][np.argmax(cv_data['test-AUC-mean'])],
#     np.argmax(cv_data['test-AUC-mean'])))
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
print(scores)
uploaded = entprise_evaluate_df.merge(test_df, on='id', how='left')
uploaded[["id", "score"]] = uploaded[["id", "score_y"]]
print(uploaded)
uploaded[["id", "score"]].to_csv("test2.csv", sep=",", index=False)
