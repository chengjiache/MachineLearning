# 导入包
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 读取数据，注意路径
train_data = pd.read_csv('train .csv')
test_data = pd.read_csv('test .csv')

# 获取训练集标签
train_label = train_data['CLASS']
del train_data['CLASS']

# 将训练集和预测集拼接一起
data_sum = pd.concat([train_data, test_data])
del data_sum['ID']

# 数据归一化
sc = StandardScaler()
sc.fit(data_sum)
data = sc.transform(data_sum)

# 分离出训练集和测试集
train_feature = data[:210, 140:242]
test_feature = data[210:, 140:242]

# 模型训练
Ir = LogisticRegression(C=100.0, random_state=1)
Ir.fit(train_feature, train_label)

# 预测
test_label = Ir.predict(test_feature)

# 输出结果
ID = range(210, 314)
df = pd.DataFrame({'ID': ID, 'CLASS': test_label})
df.to_csv("submission.csv", index=False)
