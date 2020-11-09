# coding: utf-8

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


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
change_info_cat = ["128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "907", "902", "908",
                   "903", "143", "144", "140", "150", "921",
                   "922", "925", "930", "935", "939", "172", "301", "180", "901", "190", "110", "111", "112", "113",
                   "114", "115", "116", "117", "118", "119",
                   "120", "121", "124", "125", "126"]
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
print(train_df)
train_df = pd_standardize(train_df, data2_num_columns)
test_df = pd_standardize(test_df, data2_num_columns)
all_columns = list(train_df.columns)
all_columns.remove("id")
all_columns.remove("label")
print(all_columns)
cat_columns = data2_cat_columns

from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier, Pool, datasets

cb = CatBoostClassifier(
    iterations=500,
    task_type='CPU',
    # 'rsm':0.1,
    boosting_type='Ordered',
    objective="Logloss",
    # custom_metric= ['AUC'],
    eval_metric='AUC',
    one_hot_max_size=255,
    learning_rate=0.03,
    scale_pos_weight=14,
    # 'l2_leaf_reg': 1,
    depth=8,
    max_bin=255,
    has_time=False,
    random_seed=42,
)

cat_column_indices = [train_df[all_columns].columns.get_loc(c) for c in cat_columns]

import hyperopt
from catboost import CatBoostClassifier, Pool, cv


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


from numpy.random import RandomState

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
    max_evals=100,
    trials=trials,
    rstate=RandomState(123)
)

print(best)

model = CatBoostClassifier(
    l2_leaf_reg=int(best['l2_leaf_reg']),
    learning_rate=best['learning_rate'],
    task_type='CPU',
    boosting_type='Ordered',
    objective="Logloss",
    iterations=500,
    eval_metric='AUC',
    scale_pos_weight=14,
    one_hot_max_size=255,
    loss_function='Logloss',
    random_seed=42,
    depth=best['depth'],
    max_bin=255,
    logging_level='Silent'
)

cv_data = cv(
    Pool(train_df[all_columns], train_df["label"], cat_features=cat_column_indices),
    cb.get_params(), nfold=3, verbose_eval=100
)

print('Best validation AUC score: {:.3f}±{:.3f} on step {}'.format(
    np.max(cv_data['test-AUC-mean']),
    cv_data['test-AUC-std'][np.argmax(cv_data['test-AUC-mean'])],
    np.argmax(cv_data['test-AUC-mean'])))
cb.fit(
    train_df[all_columns],
    train_df["label"],
    cat_features=cat_columns,
    # use_best_model=True,
    # eval_set=(valid_data[columns], valid_data[target]),
    # plot="True",
    verbose_eval=100,
    metric_period=100,
    early_stopping_rounds=1000,
    # snapshot_interval=1800,
    # snapshot_file="snapshot"
)
train_pool = Pool(train_df[all_columns], train_df["label"], cat_features=cat_columns)
feature_importances = cb.get_feature_importance(train_pool)
feature_names = all_columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))
res = cb.predict_proba(test_df[all_columns])
scores = np.asarray([pairs[1] for pairs in res])
test_df["score"] = scores
uploaded = entprise_evaluate_df.merge(test_df, on='id', how='left')
uploaded[["id", "score"]] = uploaded[["id", "score_y"]]
high_confidence_df = uploaded[["id", "score"]].loc[uploaded["score"] >= 0.99]
high_confidence_df["high_confidence_flag"] = 1
uploaded[["id", "score"]].to_csv("test.csv", sep=",", index=False)
