import json
import os

import numpy as np
import scipy.io as scio
from imblearn.over_sampling import SMOTE
from JSFS import JSFS

ck_name_field = ['ant1', 'ivy2', 'jedit4',
                 'lucene2', 'synapse1', 'velocity1', 'xalan2']
nasa_name_field = ['cm1', 'kc3', 'mc2', 'mw1', 'pc1', 'pc3', 'pc4', 'pc5']


def ck_benchmark():
    # run learning methods in ck dataset
    for name in ck_name_field:
        # ['ant1', 'ivy2', 'jedit4', 'lucene2', 'synapse1', 'velocity1', 'xalan2']
        """ if name not in ['velocity1']:
            continue """
        train_name = f'{name}train'
        test_name = f'{name}test'
        trian_path = f'../data/CK/CKTrain/{train_name}.mat'
        test_path = f'../data/CK/CKTest/{test_name}.mat'
        # read train dataset
        data = scio.loadmat(trian_path)
        train_divided = []
        for key in data[f'{train_name}']:
            for k in key:
                train_divided.append(k)
        # read test dataset
        data = scio.loadmat(test_path)
        test_divided = []
        for key in data[f'{test_name}']:
            for k in key:
                test_divided.append(k)

        result = {'method': "JSFS", 'dataset': 'CK', 'subDataset': name, 'result': []}

        # 10% 20% 30%
        for i in range(0, 3):
            print('\n========= CK -', name, str(10 + i * 10) + '% ==========')
            # scramble data
            np.random.shuffle(train_divided[i])
            # get 10%/20%/30% train data
            X_train = train_divided[i][:, :20]
            y_train = train_divided[i][:, 20]
            # get 90%/80%/70% test data
            X_test = test_divided[i][:, :20]
            y_test = test_divided[i][:, 20]
            # preprocessing of abnormal label value
            if y_train.mean() > 1:
                for idx, (key) in enumerate(y_train):
                    if key == 2:
                        y_train[idx] = -1.0
            if y_test.mean() > 1:
                for idx, (key) in enumerate(y_test):
                    if key == 2:
                        y_test[idx] = -1.0
            # dataset expanding by SMOTE
            smo = SMOTE(random_state=0, k_neighbors=3)
            X_train, y_train = smo.fit_resample(X_train, y_train)
            # JSFS
            result['result'].append(JSFS(X_train, y_train, X_test, y_test, name))
            result['result'][i]['percent'] = str(10 + i * 10) + '%'
        dirs = "../result/CK/JSFS/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(dirs + name + ".json", "w") as f:
            json.dump(result, f, indent=4)


def nasa_benchmark():
    # run learning methods in NASA dataset
    for name in nasa_name_field:
        limit = {'cm1': 1, 'kc3': 30, 'mc2': 40, 'mw1': 0.4, 'pc1': 1, 'pc3': 1, 'pc4': 1, 'pc5': 1}
        """ if name not in ['pc5']:
            continue """
        train_name = f'{name}train'
        test_name = f'{name}test'
        trian_path = f'../data/NASA/NASATrain/{train_name}.mat'
        test_path = f'../data/NASA/NASATest/{test_name}.mat'
        # read train dataset
        data = scio.loadmat(trian_path)
        train_divided = []
        for key in data[f'{train_name}']:
            for k in key:
                train_divided.append(k)
        # read test dataset
        data = scio.loadmat(test_path)
        test_divided = []
        for key in data[f'{test_name}']:
            for k in key:
                test_divided.append(k)

        result = {'method': "JSFS", 'dataset': 'NASA', 'subDataset': name, 'result': []}

        # 10% 20% 30%
        for i in range(0, 3):
            print('\n========= NASA -', name, str(10 + i * 10) + '% ==========')
            # scramble data
            np.random.shuffle(train_divided[i])
            # get 10%/20%/30% train data
            X_train = train_divided[i][:, :20]
            y_train = train_divided[i][:, 20]
            # get 90%/80%/70% test data
            X_test = test_divided[i][:, :20]
            y_test = test_divided[i][:, 20]
            # print(y_test)
            # preprocessing of abnormal label value
            if y_train.mean() > 1:
                for idx, (key) in enumerate(y_train):
                    if key == 2:
                        y_train[idx] = -1.0
            if y_test.mean() > 1:
                for idx, (key) in enumerate(y_test):
                    if key == 2:
                        y_test[idx] = -1.0
            # divide positive samples and negative samples
            for idx, (key) in enumerate(y_train):
                if key < limit[name]:
                    y_train[idx] = -1.0
                else:
                    y_train[idx] = 1.0
            for idx, (key) in enumerate(y_test):
                if key < limit[name]:
                    y_test[idx] = -1.0
                else:
                    y_test[idx] = 1.0
            # dataset expanding by SMOTE
            smo_1 = SMOTE(random_state=0, k_neighbors=3)
            smo_2 = SMOTE(random_state=0, k_neighbors=1)
            try:
                # X_train, y_train =  smo_1.fit_resample(X_train, y_train.astype('int'))
                X_train, y_train = smo_1.fit_resample(X_train, y_train)
            except ValueError:
                X_train, y_train = smo_2.fit_resample(X_train, y_train)
            # JSFS
            result['result'].append(JSFS(X_train, y_train, X_test, y_test, name))
            result['result'][i]['percent'] = str(10 + i * 10) + '%'
        dirs = "../result/NASA/JSFS/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(dirs + name + ".json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    ck_benchmark()
    nasa_benchmark()
