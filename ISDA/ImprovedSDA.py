import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import scipy.io
import scipy.linalg


# 函数get_skewed_F_measure用于计算偏斜F度量值
def get_skewed_F_measure(X1, X2, h1, h2, alpha):
    # 初始化真正例、真负例、假正例、假负例的数量
    TP = 0  # 预测为真实缺陷且正确的数量
    TN = 0  # 预测为非缺陷且正确的数量
    FP = 0  # 预测为非缺陷但错误的数量
    FN = 0  # 预测为真实缺陷但错误的数量

    # 遍历预测为真实缺陷的实例
    n1 = len(X1)
    for i in range(0, n1):
        y = X1[i]
        X1 = np.delete(X1, i, axis=0)  # 从X1中删除当前样本
        if get_label_of_y(X1, X2, h1, h2, y):
            TP = TP + 1  # 如果真实标签为缺陷且预测正确，则增加真正例数量
        else:
            FN = FN + 1  # 如果真实标签为缺陷但预测错误，则增加假负例数量
        X1 = np.insert(X1, i, y, axis=0)  # 将样本重新插入X1

    # 遍历预测为非缺陷的实例
    n2 = len(X2)
    for i in range(0, n2):
        y = X2[i]
        X2 = np.delete(X2, i, axis=0)  # 从X2中删除当前样本
        if get_label_of_y(X1, X2, h1, h2, y):
            FP = FP + 1  # 如果真实标签为非缺陷但预测为缺陷，则增加假正例数量
        else:
            TN = TN + 1  # 如果真实标签为非缺陷且预测正确，则增加真负例数量
        X2 = np.insert(X2, i, y, axis=0)  # 将样本重新插入X2

    # 计算精确度、召回率、假正例率、真负例率和偏斜F度量值
    Recall = TP / (TP + FN)  # 召回率
    Precision = TP / (TP + FP)  # 精确度
    Pf = FP / (FP + TN)  # 假正例率
    TNR = 1 - Pf  # 真负例率
    skewedFMeasure = (1 + alpha) * Precision * Recall / (alpha * Precision + Recall)  # 偏斜F度量值
    return skewedFMeasure  # 返回计算得到的偏斜F度量值


# 函数get_H1_H2_for_I_SDA用于获取最优的子类别大小H1和H2
def get_H1_H2_for_I_SDA(X1, X2, minSizeOfSubclass):
    # 获取源域1（X1）和源域2（X2）的样本数量
    n1 = len(X1)
    n2 = len(X2)

    # 初始化子类别大小 H1 和 H2，以及偏斜 F 度量值 skewedFMeasure
    H1 = 0
    H2 = 0
    skewedFMeasure = -1

    # 初始化子类别大小的增量 subSize
    subSize = minSizeOfSubclass

    # 循环计算不同子类别大小下的偏斜 F 度量值，并选择最佳的 H1 和 H2
    while subSize * 2 <= n1 and subSize * 2 <= n2:
        # 计算当前子类别大小对应的 H1 和 H2
        h1 = round(n1 / subSize)
        h2 = round(n2 / subSize)

        # 使用 get_skewed_F_measure 函数计算当前子类别大小下的偏斜 F 度量值 sf
        sf = get_skewed_F_measure(X1, X2, h1, h2, 4)

        # 如果当前的偏斜 F 度量值 sf 更大，则更新最佳的 H1 和 H2 以及偏斜 F 度量值 skewedFMeasure
        if skewedFMeasure < sf:
            skewedFMeasure = sf
            H1 = h1
            H2 = h2

        # 打印当前子类别大小、子类别数量以及偏斜 F 度量值 sf
        print(('h1 = %d' % h1, 'subSize1 = %d' % (n1 / h1)), ('h2 = %d' % h2, 'subSize2 = %d' % (n2 / h2)),
              'sf = %f' % sf)

        # 增加子类别大小的增量 subSize
        subSize = subSize + 1

    # 返回最佳的 H1 和 H2
    return H1, H2


# 函数sort_for_nnc用于对数据进行排序以用于最近邻分类
def sort_for_nnc(X):
    n, m = X.shape  # 获取数据矩阵 X 的形状，n 表示样本数量，m 表示特征数量
    sortedX = np.zeros(shape=(n, m))  # 创建一个与 X 相同形状的零矩阵 sortedX 用于存储排序后的样本
    euclideanDistance = np.zeros(shape=(n, n))  # 创建一个零矩阵 euclideanDistance 用于存储样本间的欧几里得距离
    maxDistance = -1  # 初始化最大距离为负无穷
    s = 0  # 初始化 s 为 0，将用于存储最小距离的样本的索引
    b = 0  # 初始化 b 为 0，将用于存储最大距离的样本的索引

    # 计算样本间的欧几里得距离并找到最大距离的样本对 (s, b)
    for i in range(0, n):
        for j in range(i + 1, n):
            euclideanDistance[i][j] = np.linalg.norm(X[i] - X[j])
            euclideanDistance[j][i] = euclideanDistance[i][j]
            if maxDistance < euclideanDistance[i][j]:
                maxDistance = euclideanDistance[i][j]
                s = i
                b = j

    # 将距离最远的两个样本放到排序后的数据矩阵的首尾
    sortedX[0] = X[s]
    sortedX[n - 1] = X[b]
    euclideanDistance[s][b] = float('inf')  # 设置 s 和 b 之间的距离为正无穷，以避免再次选择它们

    # 以下循环将剩余的样本按距离逐步排序
    for g in range(0, int((n - 1) / 2)):
        minDistance = float('inf')  # 初始化最小距离为正无穷
        m = 0  # 初始化 m 为 0，将用于存储最小距离的样本的索引

        # 找到距离 s 最近的样本，将其放到排序后的数据矩阵中
        for j in range(0, n):
            if euclideanDistance[s][j] < minDistance and j != s:
                minDistance = euclideanDistance[s][j]
                m = j
        sortedX[g + 1] = X[m]
        euclideanDistance[s][m] = float('inf')  # 设置 s 和 m 之间的距离为正无穷
        euclideanDistance[b][m] = float('inf')  # 设置 b 和 m 之间的距离为正无穷

        # 如果排序后的数据矩阵中还有样本未被处理，继续寻找最小距离的样本
        if g + 1 != n - g - 2:
            minDistance = float('inf')  # 初始化最小距离为正无穷
            k = 0  # 初始化 k 为 0，将用于存储最小距离的样本的索引

            # 找到距离 b 最近的样本，将其放到排序后的数据矩阵中
            for j in range(0, n):
                if euclideanDistance[b][j] < minDistance and j != b:
                    minDistance = euclideanDistance[b][j]
                    k = j
            sortedX[n - g - 2] = X[k]
            euclideanDistance[s][k] = float('inf')  # 设置 s 和 k 之间的距离为正无穷
            euclideanDistance[b][k] = float('inf')  # 设置 b 和 k 之间的距离为正无穷

    return sortedX  # 返回排序后的数据矩阵 sortedX


# 函数NNC用于最近邻分类
def NNC(X1, X2, H1, H2):
    # 使用 sort_for_nnc 函数对 X1 和 X2 进行排序，以用于最近邻分类
    sortedX1 = sort_for_nnc(X1)
    sortedX2 = sort_for_nnc(X2)

    # 将排序后的数据分成 H1 个子类别
    subX1 = np.array_split(sortedX1, H1)

    # 将排序后的数据分成 H2 个子类别
    subX2 = np.array_split(sortedX2, H2)

    # 返回分好的子类别
    return subX1, subX2


# 函数get_sumB用于计算矩阵sumB
def get_sumB(subX1, subX2, n1, n2):
    # 获取子类别的数量
    H1 = len(subX1)
    H2 = len(subX2)

    # 初始化 sum_B，用于计算矩阵 sumB
    sum_B = 0

    # 循环遍历子类别，计算 sumB 的各个元素
    for i in range(H1):
        # 计算子类别 i 在总体样本中的占比
        p_1i = len(subX1[i]) / n1

        # 计算子类别 i 的均值向量
        u_1i = np.mean(subX1[i], axis=0)

        # 将均值向量转换为二维数组形式
        u_1i = np.array([u_1i.tolist()])

        for j in range(H2):
            # 计算子类别 j 在总体样本中的占比
            p_2j = len(subX2[j]) / n2

            # 计算子类别 j 的均值向量
            u_2j = np.mean(subX2[j], axis=0)

            # 将均值向量转换为二维数组形式
            u_2j = np.array([u_2j.tolist()])

            # 计算均值向量之间的差距
            gap = u_1i - u_2j

            # 更新 sum_B，计算每个元素的贡献并累加
            sum_B = sum_B + (p_1i * p_2j * np.dot(gap.T, gap))

    # 返回计算得到的矩阵 sum_B
    return sum_B


# 函数get_sumX用于计算矩阵sumX
def get_sumX(X1, X2, n1, n2):
    # 计算 X1 的均值向量
    u1 = np.mean(X1, axis=0)

    # 计算 X2 的均值向量
    u2 = np.mean(X2, axis=0)

    # 计算总体样本的均值向量
    u = (u1 + u2) / 2

    # 将均值向量转换为二维数组形式
    u = np.array([u.tolist()])

    # 初始化 sum_X，用于计算矩阵 sumX
    sum_X = 0

    # 遍历 X1 中的样本，计算 sumX 的各个元素
    for i in range(n1):
        # 提取第 i 个样本
        x = np.array([X1[i].tolist()])

        # 计算样本向量与总体均值向量之间的差距
        gap = x - u

        # 更新 sum_X，计算每个元素的贡献并累加
        sum_X = sum_X + np.dot(gap, gap.T)

    # 遍历 X2 中的样本，计算 sumX 的各个元素
    for i in range(n2):
        # 提取第 i 个样本
        x = np.array([X2[i].tolist()])

        # 计算样本向量与总体均值向量之间的差距
        gap = x - u

        # 更新 sum_X，计算每个元素的贡献并累加
        sum_X = sum_X + np.dot(gap.T, gap)

    # 返回计算得到的矩阵 sum_X
    return sum_X


# 函数get_V用于计算矩阵V
def get_V(sumB, sumX):
    # 计算 sumX 的伪逆
    sumX_inv = np.linalg.pinv(sumX)

    # 计算矩阵乘积 dot = sumX_inv * sumB
    dot = np.dot(sumX_inv, sumB)

    # 计算 dot 的特征值和特征向量
    w, v = np.linalg.eig(dot)

    # 将特征向量矩阵 v 转置
    v = v.T

    # 获取特征值的长度
    len_w = len(w)

    # 初始化 V 列表
    V = []

    # 遍历特征值
    for i in range(len_w):
        # 如果特征值不为零
        if w[i] != 0:
            # 将对应的特征向量添加到 V 列表中
            V.append(v[1])

    # 将 V 列表转换为实数数组并返回
    return np.real(np.array(V))


# 函数get_label_of_y用于预测y的标签
def get_label_of_y(X1, X2, h1, h2, y):
    # 检查y的形状，如果是一维数组，则将其转换为二维数组
    if len(y.shape) == 1:
        y = np.array([y.tolist()])

    n1 = len(X1)  # 获取X1的样本数量
    n2 = len(X2)  # 获取X2的样本数量

    # Step 2: 使用 NNC 函数获取最近邻分类器的子类别
    subX1, subX2 = NNC(X1, X2, h1, h2)

    # Step 3: 计算矩阵 sum_B
    sum_B = get_sumB(subX1, subX2, n1, n2)

    # Step 4: 计算矩阵 sum_X
    sum_X = get_sumX(X1, X2, n1, n2)

    # Step 5: 计算变换矩阵 V
    V = get_V(sum_B, sum_X)

    # Step 6: 将样本数据进行特征变换
    X = np.concatenate((X1, X2), axis=0)
    X_f = np.dot(V.T, X.T)
    y_f = np.dot(V.T, y.T)

    # 为 X1 和 X2 的样本添加标签（1 表示缺陷实例，0 表示非缺陷实例）
    labels1 = np.ones(n1, dtype='i4')
    labels2 = np.zeros(n2, dtype='i4')
    labels = np.concatenate((labels1, labels2), axis=0)

    # 使用随机森林分类器对 y 进行分类预测
    rf = RandomForestClassifier()
    rf.fit(X_f.T, labels)
    predictions = rf.predict(y_f.T)

    # 返回预测结果
    return predictions


# 函数kernel用于计算核函数
def kernel(X1, X2, ker='primal', gamma=1):
    K = None

    # 如果没有指定核函数或使用 'primal' 核
    if not ker or ker == 'primal':
        # 直接返回输入的特征矩阵 X1
        K = X1

    # 如果指定核函数为 'linear' 核
    elif ker == 'linear':
        # 如果提供了第二个特征矩阵 X2
        if X2 is not None:
            # 使用线性核函数计算 X1 和 X2 之间的核矩阵
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            # 使用线性核函数计算 X1 和自身之间的核矩阵
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)

    # 如果指定核函数为 'rbf' 核
    elif ker == 'rbf':
        # 如果提供了第二个特征矩阵 X2
        if X2 is not None:
            # 使用径向基函数 (RBF) 核函数计算 X1 和 X2 之间的核矩阵，带有指定的 gamma 参数
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            # 使用径向基函数 (RBF) 核函数计算 X1 和自身之间的核矩阵，带有指定的 gamma 参数
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)

    # 返回计算得到的核矩阵 K
    return K


# 函数TCA用于域适应的特征转换
def TCA(Xs, Xt, kernel_type='primal', dim=16, lamb=1, gamma=1):
    # 将源域和目标域样本连接，并进行归一化处理
    """
    1. 将源域和目标域的特征矩阵连接成一个新的特征矩阵 X，并对特征进行归一化处理，以确保它们具有相同的尺度和分布。
    2. 计算源域和目标域样本数量以及一个权重矩阵 M，该矩阵表示样本的源域和目标域的权重关系。
    3. 计算一个中心矩阵 H，该矩阵用于保持特征在共享特征空间中的中心性。
    4. 使用指定的核函数（默认为 'primal' 核函数）计算特征矩阵 X 的核矩阵 K，并根据核函数类型选择不同的特征映射维度。
    5. 构建两个矩阵 a 和 b，这些矩阵用于执行域适应的特征转换。
    6. 使用特征值分解（EVD）计算特征矩阵 a 和 b 的特征值和特征向量。
    7. 根据特征值排序选择前 dim 个特征向量，构建一个新的特征矩阵 A。
    8. 计算最终的特征映射 Z，并对其进行归一化处理。
    9. 返回源域和目标域在共享特征空间中的新特征矩阵 Xs_new 和 Xt_new。
    """
    X = np.hstack((Xs.T, Xt.T))
    X = X / np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    M = e * e.T
    M = M / np.linalg.norm(M, 'fro')
    H = np.eye(n) - 1 / n * np.ones((n, n))
    K = kernel(X, None, ker=kernel_type, gamma=gamma)
    n_eye = m if kernel_type == 'primal' else n
    a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = V[:, ind[:dim]]
    Z = np.dot(A.T, K)
    Z = Z / np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
    return Xs_new, Xt_new


# 函数within_project_ISDA用于在同一项目中执行ISDA
def within_project_ISDA(X1, X2, y, minSizeOfSubclass):
    # 计算X1和X2的样本数量
    n1 = len(X1)
    n2 = len(X2)

    # 计算子类别的大小H1和H2，以及每个子类别的实际大小
    H1 = round(n1 / minSizeOfSubclass)
    H2 = round(n2 / minSizeOfSubclass)

    # 打印样本数量和子类别的信息
    print(('n1 = %d' % n1, 'n2 = %d' % n2))
    print(('H1 = %d' % H1, 'subSize1 = %d' % (n1 / H1)), ('H2 = %d' % H2, 'subSize2 = %d' % (n2 / H2)))

    # 使用get_label_of_y函数预测y的标签
    predictions = get_label_of_y(X1, X2, H1, H2, y)

    # 返回预测的标签
    return predictions


# 函数cross_project_SSTCA_ISDA用于跨项目的SSTCA-ISDA
def cross_project_SSTCA_ISDA(Xs1, Xs2, Xt, minSizeOfSubclass):
    # 获取源域1和源域2的样本数量
    n1 = len(Xs1)
    n2 = len(Xs2)

    # 将源域1和源域2的特征矩阵垂直堆叠，创建一个合并的源域特征矩阵 Xs
    Xs = np.vstack((Xs1, Xs2))

    # 使用 TCA 函数将合并的源域特征矩阵 Xs 和目标域特征矩阵 Xt 进行域适应的特征转换
    Xs_new, Xt_new = TCA(Xs, Xt)

    # 将转换后的特征矩阵分割回源域1和源域2
    Xs_new1 = Xs_new[: n1]
    Xs_new2 = Xs_new[n1: n1 + n2]

    # 使用 within_project_ISDA 函数在目标域上执行分类任务，返回预测结果
    predictions = within_project_ISDA(Xs_new1, Xs_new2, Xt, minSizeOfSubclass)

    # 返回在目标域上的预测结果
    return predictions


# 类ImprovedSDA用于包装ISDA算法的两种执行方式
class ImprovedSDA:
    def __init__(self, X1, X2, Xt, minSizeOfSubclass):
        """
        :param X1: n1 * n_feature, defective instances
        :param X2: n2 * n_feature, defect-free instances
        :param Xt: nt * n_feature, unlabeled instances
        :param minSizeOfSubclass:
        """
        self.X1 = X1
        self.X2 = X2
        self.Xt = Xt
        self.minSizeOfSubclass = minSizeOfSubclass

    def within_predict(self):
        return within_project_ISDA(self.X1, self.X2, self.Xt, self.minSizeOfSubclass)

    def cross_predict(self):
        return cross_project_SSTCA_ISDA(self.X1, self.X2, self.Xt, self.minSizeOfSubclass)
