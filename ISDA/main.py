from ImprovedSDA import ImprovedSDA
from DataProcess import load_data, train_data_process, test_data_process
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from Utility.print import print_result

import json
import os


def run(data_train, data_test):
    X1, X2 = train_data_process(data_train)
    Y, y_true = test_data_process(data_test)
    try:
        clf = ImprovedSDA(X1, X2, Y, minSizeOfSubclass=5)
        y_pred = clf.within_predict()
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
            'train samples': str(data_train.shape[0]),
            'defective train samples': str(X1.shape[0]),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'pf': pf,
            'F-measure': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'AUC': auc
        }
    except ZeroDivisionError as e:
        return str(e)


if __name__ == '__main__':
    NASA = ['cm1', 'kc3', 'mc2', 'mw1', 'pc1', 'pc3', 'pc4', 'pc5']
    CK = ['ant1', 'ivy2', 'jedit4', 'lucene2', 'synapse1', 'velocity1', 'xalan2']

    clf_name = 'ISDA'

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
        result['result'].append(run(data1_train, data1_test))
        result['result'].append(run(data2_train, data2_test))
        result['result'].append(run(data3_train, data3_test))
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
        result['result'].append(run(data1_train, data1_test))
        result['result'].append(run(data2_train, data2_test))
        result['result'].append(run(data3_train, data3_test))
        print_result(result)
        dirs = "../result/CK/" + clf_name + "/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(dirs + dataset + ".json", "w") as f:
            json.dump(result, f, indent=4)