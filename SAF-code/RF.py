import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

n = 20
names = ["有效碳氢比", "热值", "闪点", "溶液的碳质量分数", "沸点", "-20蒸汽压"]

name = "沸点"

def mean_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 1 - np.mean(np.abs((y_true - y_pred) / y_true))

for b in range(0, n):
    # 读取光谱数据到list变量pattern
    pattern = list()
    fhd = csv.reader(open('pattern.csv', 'r'))
    for line in fhd:
        pattern.append(line)

    # 对光谱数据进行主成分分析
    pattern = np.array(pattern, dtype='float64')  # 调整数组大小。
    pca = PCA(n_components=n)  # n_components选择主成分个数
    pca.fit(pattern)  # 训练PCA模型
    pattern = pca.transform(pattern)  # 将数据X转换成降维后的数据。
    pattern = pattern.tolist()  # 列表化

    new_data = []
    for row in pattern:
        row[b] = float(row[b])
        list1 = []
        list1.append(row[b])
        new_data.append(list1)

    # 将主成分分析结果写入新的CSV文件
    output = csv.writer(open('pattern_pca%d_pc%d.csv' % (n, b + 1), 'a', newline=''), dialect='excel')
    output.writerows(new_data)

    # 读取label数据到list变量label
    label = list()
    fhl = csv.reader(open('%s.csv' % name, 'r', encoding='utf-8-sig'))
    for line in fhl:
        label.append(line)
    label = np.array(label, dtype='float64')
    label = label.tolist()

    a = 4
    # 进行数据准备
    raw_train = list()
    label_train = list()
    raw_test = list()
    label_test = list()
    raw_cross = list()
    label_cross = list()
    i, j = 1, 1
    for line in new_data:
        if i == a:
            raw_cross.append(line)
            i += 1
        elif i == 5:
            raw_test.append(line)
            i = 1
        else:
            raw_train.append(line)
            i += 1
    for line in label:
        if j == a:
            label_cross.append(line)
            j += 1
        elif j == 5:
            label_test.append(line)
            j = 1
        else:
            label_train.append(line)
            j += 1

    # 训练RandomForestRegressor模型
    clf = RandomForestRegressor(criterion='squared_error', max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=50)
    clf.fit(raw_train, label_train)

    # 使用训练好的模型进行预测
    label_predict_cross = clf.predict(raw_cross)
    label_predict_cross = label_predict_cross.tolist()
    output = csv.writer(open('label_cross_predict_%s_%s_pca%d_pc%d_%d.csv' % (name, 'RandomForest', n, b + 1, a), 'a', newline=''), dialect='excel')
    output.writerows(map(lambda x: [x], label_predict_cross))

    label_predict_test = clf.predict(raw_test)
    label_predict_test = label_predict_test.tolist()
    output = csv.writer(open('label_predict_%s_%s_pca%d_pc%d_%d.csv' % (name, 'RandomForest', n, b + 1, a), 'a', newline=''), dialect='excel')
    output.writerows(map(lambda x: [x], label_predict_test))

    # 保存交叉验证和测试标签
    output = csv.writer(open('label_cross_%s_pca%d_pc%d_%d.csv' % (name, n, b + 1, a), 'a', newline=''), dialect='excel')
    output.writerows(map(lambda x: x, label_cross))
    output = csv.writer(open('label_test_%s_pca%d_pc%d_%d.csv' % (name, n, b + 1, a), 'a', newline=''), dialect='excel')
    output.writerows(map(lambda x: x, label_test))

    # 计算MRE和准确率
    mre_cross = mean_relative_error(label_cross, label_predict_cross)
    acc_cross = accuracy(label_cross, label_predict_cross)
    mre_test = mean_relative_error(label_test, label_predict_test)
    acc_test = accuracy(label_test, label_predict_test)

    with open('%s_MRE_results.csv' % name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PCA%d_PC%d' % (n, b + 1), 'MRE_Cross_Validation', 'MRE_Test_Set'])
        writer.writerow([b + 1, mre_cross, mre_test])

    with open('%s_Accuracy_results.csv' % name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PCA%d_PC%d' % (n, b + 1), 'Accuracy_Cross_Validation', 'Accuracy_Test_Set'])
        writer.writerow([b + 1, acc_cross, acc_test])
