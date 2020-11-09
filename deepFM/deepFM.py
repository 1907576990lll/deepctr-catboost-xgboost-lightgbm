import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import *
from deepctr.feature_column import SparseFeat, get_feature_names
import tensorflow as tf

# 数据加载
data = pd.read_csv("../../train_data_process/dataset1/train.csv")
submit_test = pd.read_csv("../../train_data_process/dataset1/test.csv")
sparse_features = ["oplocdistrict", "industryphy", "industryco", "enttype", "enttypeitem",
                   "state", "orgid", "jobid", "adbusign", "townsign", "regtype", "compform", "opform", "venind",
                   "enttypegb", "oploc"]
# sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
target = ['label']

# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
    submit_test[feature] = lbe.fit_transform(submit_test[feature])
# 计算每个特征中的 不同特征值的个数

fixlen_feature_columns = [
    SparseFeat(feature, pd.concat([data[feature], submit_test[feature]]).nunique(), embedding_dim=4) for feature in
    sparse_features]
# fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in
#                           sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print(feature_names)
# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.3)

train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

submit_test_input = {name: submit_test[name].values for name in feature_names}
# 使用DeepFM进行训练
model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary', )
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy",
              metrics=["AUC", "Precision", "Recall"], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=20, verbose=True,
                    validation_split=0.1, )
# 使用DeepFM进行预测
# 测试集效果
pred_test = model.predict(test_model_input, batch_size=256)
count = 0
pcount = 0
rcount = 0
for i in range(1487):
    if pred_test[i] > 0.5 and test["label"].values[i] == 1:
        count += 1
    if pred_test[i] > 0.5:
        pcount += 1
    if test["label"].values[i] == 1:
        rcount += 1
print("pre", count / pcount, "recall", count / rcount)

# 提交的文件
print(len(test_model_input), test_model_input["industryco"].shape, test_model_input)
print(len(submit_test_input), submit_test_input["industryco"].shape, submit_test_input)
pred_ans = model.predict(submit_test_input, batch_size=256)
df = pd.DataFrame(columns=["id", "score"], data=None)
df["id"] = submit_test["id"].values
df["score"] = pred_ans
print(df)
df.to_csv("test_DFM.csv")
# 两个测试集效果差异很大
# for i in test_model_input:
#     print(i,len(set(train_model_input[i].tolist())))
#     print(i,len(set(test_model_input[i].tolist())))
#     print(i,len(set(submit_test_input[i].tolist())))
