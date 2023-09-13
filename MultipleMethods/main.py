from pyod.models.cblof import CBLOF
from Utility.print import print_result

from DataProcess import load_data, train_data_process, test_data_process
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from pyod.models.xgbod import XGBOD
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

import json
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# 完全禁用警告
warnings.filterwarnings("ignore")


def run(data_train, data_test, clf_name):
    X_train, y_train = train_data_process(data_train)
    X_test, y_true = test_data_process(data_test)
    classifiers = {
        # 有监督
        # XGBOD是一种使用XGBoost算法的异常检测模型。它使用梯度提升树方法来识别数据中的异常值。XGBoost是一个强大的集成学习算法，它在处理大规模数
        # 据和复杂特征时表现出色
        "XGBOD": XGBOD(),

        # K最近邻分类器是一种基于邻居的分类算法。它使用训练数据中最接近一个数据点的K个最近邻居的标签来预测该数据点的标签。这是一种简单而有效的分类
        # 算法
        "KNeighborsClassifier": KNeighborsClassifier(3),

        # 支持向量分类器是一种基于支持向量机的分类算法。它通过构建一个超平面来分隔不同类别的数据点，并尽量使间隔最大化，从而进行分类。
        "SVC": SVC(random_state=0),

        # 高斯过程分类器是一种基于高斯过程的概率模型，用于分类任务。它通过学习训练数据的概率分布来进行分类预测。
        "GaussianProcessClassifier": GaussianProcessClassifier(1.0 * RBF(1.0)),

        # 决策树分类器是一种基于树结构的分类算法。它通过将数据分成不同的决策节点来进行分类决策，每个节点代表一个特征，并且根据这些特征进行分类。
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),

        # 随机森林分类器是一种集成学习算法，它使用多个决策树来进行分类，并通过投票或平均来得出最终的分类结果。它通常具有很好的性能和鲁棒性。
        "RandomForestClassifier": RandomForestClassifier(random_state=0),

        # 多层感知器分类器是一种人工神经网络模型，用于分类任务。它由多个神经元层组成，可以处理复杂的非线性关系。
        "MLPClassifier": MLPClassifier(random_state=0),

        # AdaBoost分类器是一种集成学习算法，它使用多个弱分类器来进行分类，并通过加权投票来得出最终的分类结果。它通常具有很好的性能和鲁棒性。
        "AdaBoostClassifier": AdaBoostClassifier(),

        # 高斯朴素贝叶斯分类器是一种基于贝叶斯定理的概率模型，用于分类任务。它假设所有特征都是相互独立的，并且每个特征都服从高斯分布。
        "GaussianNB": GaussianNB(),

        # 二次判别分析是一种基于贝叶斯定理的概率模型，用于分类任务。它假设所有特征都服从高斯分布，并且每个类别都有自己的协方差矩阵。
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),

        # 无监督
        # 基于簇的局部离群因子是一种异常检测算法，它通过将数据点分为簇并计算每个数据点的局部离群因子来识别异常值。CBLOF特别适用于集群结构的数据。
        "CBLOF": CBLOF(random_state=0)
    }

    clf = classifiers[clf_name]

    try:
        if clf_name == "CBLOF":
            clf.fit(X_train)
        else:
            clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for i, label in enumerate(y_true):
            if label:
                if y_pred[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if y_pred[i]:
                    FP += 1
                else:
                    TN += 1
        if (FP + TN) == 0:
            pf = "no negative samples."
        else:
            pf = FP / (FP + TN)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError as e:
            auc = str(e)
        return {
            'train samples': str(X_train.shape[0]),
            'defective train samples': str(np.sum(y_train)),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'pf': pf,
            'F-measure': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'AUC': auc
        }
    except ValueError as e:
        return str(e)


if __name__ == '__main__':
    NASA = ['cm1', 'kc3', 'mc2', 'mw1', 'pc1', 'pc3', 'pc4', 'pc5']
    CK = ['ant1', 'ivy2', 'jedit4', 'lucene2', 'synapse1', 'velocity1', 'xalan2']

    clf_name = 'CBLOF'

    for dataset in NASA:
        data_name_train = dataset + 'train'
        filepath_train = '../data/NASA/NASATrain/' + data_name_train + '.mat'
        data1_train, data2_train, data3_train = load_data(filepath_train, data_name_train)
        data_name_test = dataset + 'test'
        filepath_test = '../data/NASA/NASATest/' + data_name_test + '.mat'
        data1_test, data2_test, data3_test = load_data(filepath_test, data_name_test)
        result = {
            'method': clf_name,
            'dataset': 'NASA',
            'subDataset': dataset,
            'result': []
        }
        result['result'].append(run(data1_train, data1_test, clf_name))
        result['result'].append(run(data2_train, data2_test, clf_name))
        result['result'].append(run(data3_train, data3_test, clf_name))
        print_result(result)
        dirs = "../result/NASA/" + clf_name + "/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(dirs + dataset + ".json", "w") as f:
            json.dump(result, f, indent=4)

    for dataset in CK:
        data_name_train = dataset + 'train'
        filepath_train = '../data/CK/CKTrain/' + data_name_train + '.mat'
        data1_train, data2_train, data3_train = load_data(filepath_train, data_name_train)
        data_name_test = dataset + 'test'
        filepath_test = '../data/CK/CKTest/' + data_name_test + '.mat'
        data1_test, data2_test, data3_test = load_data(filepath_test, data_name_test)
        result = {
            'method': clf_name,
            'dataset': 'CK',
            'subDataset': dataset,
            'result': []
        }
        result['result'].append(run(data1_train, data1_test, clf_name))
        result['result'].append(run(data2_train, data2_test, clf_name))
        result['result'].append(run(data3_train, data3_test, clf_name))
        print_result(result)
        dirs = "../result/CK/" + clf_name + "/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(dirs + dataset + ".json", "w") as f:
            json.dump(result, f, indent=4)
