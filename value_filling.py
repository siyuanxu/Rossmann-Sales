import numpy as np
import pandas as pd

train = pd.read_csv('data/good_train_with_nan.csv')
# filling values

cd_fill_value = np.mean([i for i in train.CompetitionDistance.values if not pd.isnull(i)])
ct_fill_value = -12
is_promo2_a_value = 0
is_promo2_b_value = 0
is_promo2_c_value = 0
p2t_fill_value = np.mean([i for i in train.Promo2Time.values if i<=0])
# 将确定好的填充值添加到数据集里

## 定义填充器
cd_filler = lambda x: cd_fill_value if pd.isnull(x) else x
ct_filler = lambda x: ct_fill_value if pd.isnull(x) else x
is_promo2_a_filler = lambda x: is_promo2_a_value if pd.isnull(x) else x
is_promo2_b_filler = lambda x: is_promo2_b_value if pd.isnull(x) else x
is_promo2_c_filler = lambda x: is_promo2_c_value if pd.isnull(x) else x
p2t_fill_filler = lambda x: p2t_fill_value if pd.isnull(x) else x

## 应用填充
train.CompetitionDistance = train.CompetitionDistance.apply(cd_filler)
train.CompetitionTime = train.CompetitionTime.apply(ct_filler)
train.is_promo2_a = train.is_promo2_a.apply(is_promo2_a_filler)
train.is_promo2_b = train.is_promo2_b.apply(is_promo2_b_filler)
train.is_promo2_c = train.is_promo2_c.apply(is_promo2_c_filler)
train.Promo2Time = train.Promo2Time.apply(p2t_fill_filler)

test = pd.read_csv('data/good_test_with_nan.csv')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 筛选训练集里的必要特征
open_X = train[['Store','is_state_holiday','SchoolHoliday','DayOfWeek']]
open_y = train.Open
# 将测试集里缺失数据的部分和完整部分分割开来
test_no_nan = test[pd.notnull(test.Open)]
# 建立分类器并
X_train, X_test, y_train, y_test = train_test_split(open_X, open_y, test_size=0.01, random_state=42)
clf = RandomForestClassifier().fit(X_train, y_train)

test_open_pred = clf.predict(test[['Store','is_state_holiday','SchoolHoliday','DayOfWeek']])

test_open_filled = []

for i in range(len(test_open_pred)):
    if pd.isnull(test.Open.values[i]):
        test_open_filled.append(test_open_pred[i])
    else:test_open_filled.append(test.Open.values[i])

test.Open = test_open_filled

## 应用填充
test.CompetitionDistance = test.CompetitionDistance.apply(cd_filler)
test.CompetitionTime = test.CompetitionTime.apply(ct_filler)
test.is_promo2_a = test.is_promo2_a.apply(is_promo2_a_filler)
test.is_promo2_b = test.is_promo2_b.apply(is_promo2_b_filler)
test.is_promo2_c = test.is_promo2_c.apply(is_promo2_c_filler)
test.Promo2Time = test.Promo2Time.apply(p2t_fill_filler)

train.to_csv('data/good_train_filled.csv')
test.to_csv('data/good_test_filled.csv')