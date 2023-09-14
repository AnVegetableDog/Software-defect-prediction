from scipy.io import loadmat
import numpy as np


# 该函数使用 load-mat 函数从MATLAB格式的文件中加载数据，然后返回数据的三个部分
def load_data(filepath, data_name):
    data = loadmat(filepath)[data_name]
    # data[:, 0][0] 返回第一列的第一个元素。
    # data[:, 1][0] 返回第二列的第一个元素。
    # data[:, 2][0] 返回第三列的第一个元素。
    return data[:, 0][0], data[:, 1][0], data[:, 2][0]


def train_data_process(data):
    """
    对训练数据进行预处理，将其分为两个子集 X1 和 X2。

    参数:
    data (numpy.ndarray): 包含训练数据的二维NumPy数组，每行表示一个样本，最后一列是标签。

    返回:
    X1 (numpy.ndarray): 包含正标签样本的特征的二维NumPy数组。
    X2 (numpy.ndarray): 包含负标签样本的特征的二维NumPy数组。
    """
    n, m = data.shape  # 获取数据集的行数和列数
    X1 = np.empty(shape=(0, m - 1))  # 创建一个空的NumPy数组，用于存储正标签样本的特征
    X2 = np.empty(shape=(0, m - 1))  # 创建一个空的NumPy数组，用于存储负标签样本的特征
    for item in data:
        if item[m - 1] > 0:
            X1 = np.row_stack((X1, item[0:m - 1]))  # 将正标签样本的特征添加到X1中
        else:
            X2 = np.row_stack((X2, item[0:m - 1]))  # 将负标签样本的特征添加到X2中
    return X1, X2  # 返回正标签和负标签样本的特征集合


def test_data_process(data):
    """
    对测试数据进行预处理，提取特征和标签。

    参数:
    data (numpy.ndarray): 包含测试数据的二维NumPy数组，每行表示一个样本，最后一列是标签。

    返回:
    X (numpy.ndarray): 包含测试样本的特征的二维NumPy数组。
    y (numpy.ndarray): 包含测试样本的标签的NumPy数组，标签为0或1。
    """
    n, m = data.shape  # 获取数据集的行数和列数
    X = data[:, 0:m - 1]  # 提取特征，即除最后一列之外的所有列
    y = (data[:, m - 1] > 0).astype(int)  # 提取标签，并将标签转换为0或1
