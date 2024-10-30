import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import csv
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

# 代码说明：需要准备一个原始光谱数据文件pattern.csv，原始label文件label.csv，代码最后会输出一个PCA结果文件pattern_pca.csv以及SVR预测结果文件label_predict.pca

n = 20
names = ["有效碳氢比", "热值", "闪点", "溶液的碳质量分数", "沸点", "溶液的氢质量分数"]
name = "溶液的氢质量分数"

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
    pca.fit(pattern)  # fit()可以说是scikit-learn中通用的方法，每个需要训练的算法都会有fit()方法，它其实就是算法中的“训练”这一步骤。
    pattern = pca.transform(pattern)  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
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
    # 进行SVR数据准备
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
            i = i + 1
        elif i == 5:
            raw_test.append(line)
            i = 1
        else:
            raw_train.append(line)
            i = i + 1
    for line in label:
        if j == a:
            label_cross.append(line)
            j = j + 1
        elif j == 5:
            label_test.append(line)
            j = 1
        else:
            label_train.append(line)
            j = j + 1

    # 进行SVR并将预测结果写入新的CSV文件
    mods = ['linear', 'rbf', 'poly']
    for mod in mods:
        clf = svm.SVR(kernel=mod)  # kernel变量分别为linear、rbf、poly
        clf.fit(raw_train, np.ravel(label_train))  # 注意label_train的转置
        label_predict_cross = clf.predict(raw_cross)
        label_predict_cross = label_predict_cross.tolist()
        label_predict_test = clf.predict(raw_test)
        label_predict_test = label_predict_test.tolist()

        output = csv.writer(open('label_predict_%s_%s_pca%d_pc%d_交叉训练集%d.csv' % (name, mod, n, b + 1, a), 'a', newline=''), dialect='excel')
        for true_val, pred_val in zip(label_cross, label_predict_cross):
            output.writerow([true_val[0], pred_val])

        for true_val, pred_val in zip(label_test, label_predict_test):
            output.writerow([true_val[0], pred_val])

        # 计算MRE和准确率
        mre_cross = mean_relative_error(label_cross, label_predict_cross)
        acc_cross = accuracy(label_cross, label_predict_cross)
        mre_test = mean_relative_error(label_test, label_predict_test)
        acc_test = accuracy(label_test, label_predict_test)

        with open('%s_%s_MRE_results.csv' % (name, mod), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['PCA%d_PC%d' % (n, b + 1), 'MRE_Cross_Validation', 'MRE_Test_Set'])
            writer.writerow([b + 1, mre_cross, mre_test])

        with open('%s_%s_Accuracy_results.csv' % (name, mod), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['PCA%d_PC%d' % (n, b + 1), 'Accuracy_Cross_Validation', 'Accuracy_Test_Set'])
            writer.writerow([b + 1, acc_cross, acc_test])
