import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import csv
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

n = 20

names = ["有效碳氢比", "热值", "闪点", "溶液的氢质量分数", "溶液的碳质量分数","沸点"]

name = "有效碳氢比"

mods = ['lbfgs']
for mod in mods:

    moods = ['identity', 'logistic']
    for mood in moods:

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
            pattern_transformed = pca.transform(pattern)  # 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
            pattern_transformed = pattern_transformed.tolist()  # 列表化

            new_data = []
            for row in pattern_transformed:
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
            for line in pattern_transformed:
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

            # 模型训练与预测
            clf = MLPRegressor(solver=mod, activation=mood, hidden_layer_sizes=(5, 5), random_state=None,
                               learning_rate_init=0.001)
            clf.fit(raw_train, np.ravel(label_train))  # 注意label_train的转置

            label_predict_cross = clf.predict(raw_cross)
            label_predict_test = clf.predict(raw_test)

            # 计算MRE
            mre_cross = np.mean(
                np.abs(label_predict_cross - np.ravel(label_cross)) / np.abs(np.ravel(label_cross))) * 100
            mre_test = np.mean(np.abs(label_predict_test - np.ravel(label_test)) / np.abs(np.ravel(label_test))) * 100

            print("MRE for Cross Validation (PCA%d PC%d): %.2f%%" % (n, b + 1, mre_cross))
            print("MRE for Test Set (PCA%d PC%d): %.2f%%" % (n, b + 1, mre_test))

            # 写入结果文件
            with open('MRE_results.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['PCA%d_PC%d' % (n, b + 1), 'MRE_Cross_Validation', 'MRE_Test_Set'])
                writer.writerow([b + 1, mre_cross, mre_test])

            with open('label_cross_%s_%s_%s_pca%d_pc%d.csv' % (name, mod, mood, n, b+1), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(label_cross)

            with open('label_test_%s_%s_%s_pca%d_pc%d.csv' % (name, mod, mood, n, b+1), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(label_test)

