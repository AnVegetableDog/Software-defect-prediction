

from __future__ import division

import itertools
import numbers
import numpy as np
from warnings import warn
from abc import ABCMeta, abstractmethod

from sklearn.base import ClassifierMixin, RegressorMixin
# from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
# from sklearn.externals.six import with_metaclass
# from sklearn.externals.six.moves import zip
from six import with_metaclass
from six.moves import zip
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils import indices_to_mask, check_consistent_length
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets

# from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble._base import _partition_estimators

__all__ = ["BaggingClassifierPU"]

MAX_INT = np.iinfo(np.int32).max


# 用于生成随机样本索引，根据是否使用自助抽样（bootstrap）来选择样本。
# 生成样本索引的函数，根据是否使用自助抽样（bootstrap）来选择样本。
def _generate_indices(random_state, bootstrap, n_population, n_samples):
    if bootstrap:
        # 如果使用自助抽样，从总体中随机抽取样本索引
        indices = random_state.randint(0, n_population, n_samples)
    else:
        # 如果不使用自助抽样，使用sample_without_replacement函数来随机选择不重复的样本索引
        indices = sample_without_replacement(n_population, n_samples, random_state=random_state)

    return indices


# 生成用于构建每个基本估计器的随机特征和样本索引的函数。对于PU学习，确保包含一组平衡的正例（P）和未标记（U）样本。
def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    random_state = check_random_state(random_state)

    # 绘制特征索引
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features)
    # 绘制样本索引
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples)

    return feature_indices, sample_indices


# 用于在并行作业中构建一组基本估计器，确保每个包中都有平衡的正例（P）和未标记（U）样本。
def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is None:
        raise ValueError("The base estimator doesn't support sample weight")

    # 存储构建的估计器和相关的特征索引
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        # 为当前估计器创建一个随机数种子
        random_state = np.random.RandomState(seeds[i])
        # 创建一个新的基本估计器实例
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # 找到正例（P）和未标记（U）的样本索引
        iP = [pair[0] for pair in enumerate(y) if pair[1] == 1]
        iU = [pair[0] for pair in enumerate(y) if pair[1] < 1]
        # 生成随机特征和样本索引
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      len(iU), max_features,
                                                      max_samples)
        # 组合正例（P）和未标记（U）的索引，以确保每个包中都包含平衡的样本
        indices = [iU[i] for i in indices] + iP

        # 根据是否支持样本权重来拟合基本估计器
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                # 计算自助抽样的样本计数
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                # 对于不使用自助抽样，将不在索引中的样本的权重设置为0
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)
        else:
            estimator.fit((X[indices])[:, features], y[indices])

        # 将构建的估计器和相关的特征索引存储在列表中
        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features

# 用于在并行作业中计算基本估计器的（概率）预测。
def _parallel_predict_proba(estimators, estimators_features, X, n_classes):
    # 获取输入数据的样本数
    n_samples = X.shape[0]
    # 创建一个用于存储概率预测的数组，维度为 (样本数, 类别数)
    proba = np.zeros((n_samples, n_classes))

    # 遍历每个基本估计器和其对应的特征索引
    for estimator, features in zip(estimators, estimators_features):
        # 检查估计器是否支持 predict_proba 方法
        if hasattr(estimator, "predict_proba"):
            # 使用基本估计器进行概率预测
            proba_estimator = estimator.predict_proba(X[:, features])

            # 检查预测的类别数是否与基本估计器的类别数一致
            if n_classes == len(estimator.classes_):
                # 如果一致，直接将概率预测累加到 proba 数组中
                proba += proba_estimator
            else:
                # 如果不一致，根据基本估计器的类别重新映射概率预测
                proba[:, estimator.classes_] += \
                    proba_estimator[:, range(len(estimator.classes_))]
        else:
            # 如果基本估计器不支持 predict_proba 方法，使用投票策略
            predictions = estimator.predict(X[:, features])

            # 遍历每个样本，将预测的类别累加到 proba 数组中
            for i in range(n_samples):
                proba[i, predictions[i]] += 1

    return proba


# 用于在并行作业中计算基本估计器的对数概率预测。
def _parallel_predict_log_proba(estimators, estimators_features, X, n_classes):
    # 获取输入数据的样本数
    n_samples = X.shape[0]
    # 创建一个用于存储对数概率预测的数组，初始化为负无穷
    log_proba = np.empty((n_samples, n_classes))
    log_proba.fill(-np.inf)
    # 创建一个包含所有可能类别的数组
    all_classes = np.arange(n_classes, dtype=np.int)

    # 遍历每个基本估计器和其对应的特征索引
    for estimator, features in zip(estimators, estimators_features):
        # 使用基本估计器进行对数概率预测
        log_proba_estimator = estimator.predict_log_proba(X[:, features])

        # 检查预测的类别数是否与基本估计器的类别数一致
        if n_classes == len(estimator.classes_):
            # 如果一致，使用对数概率的对数加法更新 log_proba 数组
            log_proba = np.logaddexp(log_proba, log_proba_estimator)
        else:
            # 如果不一致，根据基本估计器的类别重新映射对数概率
            log_proba[:, estimator.classes_] = np.logaddexp(
                log_proba[:, estimator.classes_],
                log_proba_estimator[:, range(len(estimator.classes_))])

            # 找到缺失的类别，并将其对数概率设置为负无穷
            missing = np.setdiff1d(all_classes, estimator.classes_)
            log_proba[:, missing] = np.logaddexp(log_proba[:, missing],
                                                 -np.inf)

    return log_proba
# 用于在并行作业中计算基本估计器的决策函数。
def _parallel_decision_function(estimators, estimators_features, X):
    # 初始化一个变量用于存储决策函数的累加值
    decision_function_sum = 0

    # 遍历每个基本估计器以及其对应的特征索引
    for estimator, features in zip(estimators, estimators_features):
        # 使用基本估计器的 decision_function 方法计算决策函数并将结果累加到 decision_function_sum
        decision_function_sum += estimator.decision_function(X[:, features])

    return decision_function_sum


# 这是一个抽象基类，定义了BaggingClassifierPU的核心逻辑。
# 包括初始化方法、fit方法、_fit方法、_set_oob_score方法等。
# _fit方法用于构建Bagging集成。
# _set_oob_score方法用于计算袋外（out-of-bag）评估分数。
class BaseBaggingPU(with_metaclass(ABCMeta, BaseEnsemble)):

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=True,
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        # 调用父类构造函数，初始化基本估计器和基本估计器数量
        super(BaseBaggingPU, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators)

        # 设置 Bagging 算法的参数
        self.max_samples = max_samples  # 最大采样样本数
        self.max_features = max_features  # 最大采样特征数
        self.bootstrap = bootstrap  # 是否使用自助抽样
        self.bootstrap_features = bootstrap_features  # 是否对特征进行自助抽样
        self.oob_score = oob_score  # 是否计算袋外评估分数
        self.warm_start = warm_start  # 是否启用热启动，即在已有估计器的基础上继续训练
        self.n_jobs = n_jobs  # 并行训练时使用的 CPU 核数
        self.random_state = random_state  # 随机数生成器的种子
        self.verbose = verbose  # 控制输出详细程度的参数

    # 用于拟合 Bagging 集成模型。
    def fit(self, X, y, sample_weight=None):
        # 调用 _fit 方法，将 max_samples 参数设置为 self.max_samples
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        # 随机状态生成器
        random_state = check_random_state(self.random_state)

        # 存储目标变量 y
        self.y = y

        # 将输入数据 X 和目标变量 y 转换为合适的格式
        X, y = check_X_y(X, y, ['csr', 'csc'])

        # 检查样本权重
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # 重新映射目标变量
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # 检查参数
        self._validate_estimator()

        # 如果指定了 max_depth，则将基本估计器的 max_depth 属性设置为指定的值
        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # 验证 max_samples 参数
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * sum(y < 1))

        if not (0 < max_samples <= sum(y < 1)):
            raise ValueError("max_samples must be positive"
                             " and no larger than the number of unlabeled points")

        # 存储经验证的整数样本采样值
        self._max_samples = max_samples

        # 验证 max_features 参数
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # 存储经验证的整数特征采样值
        self._max_features = max_features

        # 其他检查
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # 释放分配的内存，如果有的话
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # 并行处理
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # 将随机状态前进到训练第一个 n_estimators 后的状态
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        # 生成一组随机种子
        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        # 并行训练基本估计器
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # 汇总结果
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        # 如果启用了 oob_score，则计算 oob 分数
        if self.oob_score:
            self._set_oob_score(X, y)

        return self

    # 计算袋外（out-of-bag）评估分数
    @abstractmethod
    def _set_oob_score(self, X, y):
        """Calculate out of bag predictions and score."""

    def _validate_y(self, y):
        # 默认实现
        return column_or_1d(y, warn=True)

    def _get_estimators_indices(self):
        # 获取在样本和特征两个轴上绘制的索引
        for seed in self._seeds:
            # 访问 random_state 的操作必须与 `_parallel_build_estimators()` 中的操作完全一致
            random_state = np.random.RandomState(seed)

            # 获取正例 (1) 和未标记 (<1) 样本的索引
            iP = [pair[0] for pair in enumerate(self.y) if pair[1] == 1]
            iU = [pair[0] for pair in enumerate(self.y) if pair[1] < 1]

            # 生成特征和样本索引
            feature_indices, sample_indices = _generate_bagging_indices(
                random_state, self.bootstrap_features, self.bootstrap,
                self.n_features_, len(iU), self._max_features,
                self._max_samples)

            # 重新排列样本索引，将未标记样本（U）放在前面，正例样本（P）放在后面
            sample_indices = [iU[i] for i in sample_indices] + iP

            yield feature_indices, sample_indices

    @property
    def estimators_samples_(self):

        sample_masks = []
        for _, sample_indices in self._get_estimators_indices():
            # 将样本索引转换为掩码
            mask = indices_to_mask(sample_indices, self._n_samples)
            sample_masks.append(mask)

        return sample_masks


# 这是BaseBaggingPU的具体实现，继承了BaseBaggingPU和ClassifierMixin。
# 初始化方法：设置Bagging算法的各种参数，如基本估计器、基本估计器数量、采样参数等。
class BaggingClassifierPU(BaseBaggingPU, ClassifierMixin):

    def __init__(self,
                 base_estimator=None,  # 基础估计器，默认为None，可以是任何机器学习模型。
                 n_estimators=10,  # 集成中的基本估计器数量，默认为10。
                 max_samples=1.0,  # 每个基本估计器的最大样本数，默认为1.0，表示使用全部样本。
                 max_features=1.0,  # 每个基本估计器的最大特征数，默认为1.0，表示使用全部特征。
                 bootstrap=True,  # 是否启用样本的自助采样，默认为True。
                 bootstrap_features=False,  # 是否启用特征的自助采样，默认为False。
                 oob_score=True,  # 是否计算袋外（out-of-bag）评估分数，默认为True。
                 warm_start=False,  # 是否启用热启动（warm start），默认为False。
                 n_jobs=1,  # 用于拟合和预测的并行作业数量，默认为1。
                 random_state=None,  # 随机数生成器的种子，用于控制伪随机性，默认为None。
                 verbose=0):  # 控制拟合过程中的冗长度级别，默认为0，不显示冗长信息。
        super(BaggingClassifierPU, self).__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    # 检查并设置基本估计器。
    def _validate_estimator(self):
        """检查估计器并设置 base_estimator_ 属性。

        如果未提供基本估计器，则将默认估计器设置为 DecisionTreeClassifier。
        """
        super(BaggingClassifierPU, self)._validate_estimator(
            default=DecisionTreeClassifier())

    # 计算袋外（out-of-bag）评估分数。
    def _set_oob_score(self, X, y):
        """计算袋外评估分数。

        Parameters:
        X : array-like, shape (n_samples, n_features)
            输入特征数据。

        y : array-like, shape (n_samples,)
            输入目标数据。

        """
        n_samples = y.shape[0]  # 获取样本数
        n_classes_ = self.n_classes_  # 获取类别数
        classes_ = self.classes_  # 获取类别标签

        predictions = np.zeros((n_samples, n_classes_))  # 初始化预测矩阵

        # 遍历每个基本估计器，计算其在袋外样本上的预测
        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # 创建用于OOB样本的掩码
            mask = ~samples

            if hasattr(estimator, "predict_proba"):
                # 如果基本估计器支持概率预测，则使用 predict_proba 方法
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                # 否则，使用 predict 方法，并为每个样本记录预测
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        # 修改：不会有关于非-OOB点（即正例）的警告
        with np.errstate(invalid='ignore'):
            # 计算袋外评估分数和决策函数
            oob_decision_function = (predictions /
                                     predictions.sum(axis=1)[:, np.newaxis])
            oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        # 存储袋外评估分数
        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score

    # 用于验证输入标签 y 是否合法
    def _validate_y(self, y):
        """验证输入标签 y 是否合法。

        Parameters:
        y : array-like, shape (n_samples,)
            输入目标数据。

        Returns:
        y : array-like, shape (n_samples,)
            经过验证后的目标数据。
        """
        y = column_or_1d(y, warn=True)  # 将 y 转换为一维数组
        check_classification_targets(y)  # 检查分类目标
        self.classes_, y = np.unique(y, return_inverse=True)  # 获取类别并将 y 转换为类别的索引
        self.n_classes_ = len(self.classes_)  # 获取类别数

        return y

    # 进行预测，返回样本的类别标签。
    def predict(self, X):
        """进行预测，返回样本的类别标签。

        Parameters:
        X : array-like, shape (n_samples, n_features)
            输入特征数据。

        Returns:
        y_pred : array-like, shape (n_samples,)
            预测的类别标签。
        """
        predicted_probability = self.predict_proba(X)  # 获取预测的类别概率
        return self.classes_.take((np.argmax(predicted_probability, axis=1)),
                                  axis=0)  # 返回具有最高概率的类别标签

    # 预测类别的概率
    def predict_proba(self, X):
        """预测类别的概率。

        Parameters:
        X : array-like, shape (n_samples, n_features)
            输入特征数据。

        Returns:
        proba : array-like, shape (n_samples, n_classes)
            预测的类别概率。
        """
        check_is_fitted(self, "classes_")  # 检查模型是否已经拟合
        X = check_array(X, accept_sparse=['csr', 'csc'])  # 检查输入数据
        if self.n_features_ != X.shape[1]:
            raise ValueError("模型的特征数量必须与输入数据匹配。模型的特征数量为 {0}，输入数据的特征数量为 {1}。"
                             "".format(self.n_features_, X.shape[1]))

        # 并行循环，获取每个基本估计器的类别概率
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_)
            for i in range(n_jobs))

        # 汇总概率
        proba = sum(all_proba) / self.n_estimators

        return proba

    # 预测类别的对数概率
    def predict_log_proba(self, X):
        """预测类别的对数概率。

        Parameters:
        X : array-like, shape (n_samples, n_features)
            输入特征数据。

        Returns:
        log_proba : array-like, shape (n_samples, n_classes)
            预测的类别的对数概率。
        """
        check_is_fitted(self, "classes_")  # 检查模型是否已经拟合

        if hasattr(self.base_estimator_, "predict_log_proba"):
            # 如果基本估计器支持预测类别的对数概率
            X = check_array(X, accept_sparse=['csr', 'csc'])  # 检查输入数据

            if self.n_features_ != X.shape[1]:
                raise ValueError("模型的特征数量必须与输入数据匹配。模型的特征数量为 {0}，输入数据的特征数量为 {1}。"
                                 "".format(self.n_features_, X.shape[1]))

            # 并行循环，获取每个基本估计器的类别对数概率
            n_jobs, n_estimators, starts = _partition_estimators(
                self.n_estimators, self.n_jobs)
            all_log_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_log_proba)(
                    self.estimators_[starts[i]:starts[i + 1]],
                    self.estimators_features_[starts[i]:starts[i + 1]],
                    X,
                    self.n_classes_)
                for i in range(n_jobs))

            # 汇总对数概率
            log_proba = all_log_proba[0]

            for j in range(1, len(all_log_proba)):
                log_proba = np.logaddexp(log_proba, all_log_proba[j])

            log_proba -= np.log(self.n_estimators)

            return log_proba

        else:
            # 如果基本估计器不支持预测类别的对数概率，则计算类别的对数概率
            return np.log(self.predict_proba(X))

    # 返回平均的基本估计器的决策函数。
    @if_delegate_has_method(delegate='base_estimator')
    def decision_function(self, X):
        """返回平均的基本估计器的决策函数。

        Parameters:
        X : array-like, shape (n_samples, n_features)
            输入特征数据。

        Returns:
        decisions : array-like, shape (n_samples, n_classes)
            平均的基本估计器的决策函数。
        """
        check_is_fitted(self, "classes_")  # 检查模型是否已经拟合

        # 检查输入数据
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("模型的特征数量必须与输入数据匹配。模型的特征数量为 {0}，输入数据的特征数量为 {1}。"
                             "".format(self.n_features_, X.shape[1]))

        # 并行循环，获取每个基本估计器的决策函数
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))

        # 汇总决策函数
        decisions = sum(all_decisions) / self.n_estimators

        return decisions
