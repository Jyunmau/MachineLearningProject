import data_process as dp
import numpy as np


# _*_ coding:utf-8 _*_

#   @Version : 0.1.0
#   @Time    : 2019/10/22 16:23
#   @Author  : Jyunmau Chan
#   @File    : logistic.py


class LogisticRegression:
    """logistic回归"""

    def __init__(self, learning_rate, batch_size, max_iter):
        """
        初始化模型参数
        :param learning_rate: 学习率
        :param batch_size: mini-batch的大小
        :param max_iter: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.sample_num = None
        self.feature_num = None
        self.category_num = None
        self.theta = None
        self.loss_list = []

    def train(self, x_train, y_train, y, is_SGD=False):
        """
        用mini-batch的GD训练并计算loss
        :param x_train: 特征数据，dim1是样本，dim2是特征
        :param y_train: 分类真实值，one_hot，dim1是样本，dim2是类别
        :return:
        """
        self.sample_num, self.feature_num = x_train.shape
        # self.sample_num, self.category_num = y_train.shape
        self.category_num = 1
        # self.theta = np.random.rand(self.category_num, self.feature_num)
        self.theta = np.zeros((self.category_num, self.feature_num))
        print(self.theta.shape)
        batches = self.get_batches(x_train, y_train, is_SGD)
        for i in range(self.max_iter):
            # batches = self.get_batches(x_train, y_train, True)
            for batch in batches:
                x_batch = batch[0]
                y_batch = batch[1]
                h_j = self.logistic(x_batch)
                d_theta = self.gradient(x_batch, y_batch, h_j)
                self.theta = self.theta - self.learning_rate * d_theta.T
            loss = self.loss(x_train, y_train)
            # print(loss)
            self.loss_list.append(loss)
        pd = dp.PlotData()
        pd.plot_loss(self.loss_list)
        pd.plot_result(x_train, y, self.theta)

    def get_batches(self, x_train, y_train, is_shuffle=False):
        """
        将数据划分成batch
        :param x_train: 特征数据，dim1是样本，dim2是特征
        :param y_train: 分类真实值，one_hot，dim1是样本，dim2是类别
        :param is_shuffle: 是否要打乱特征数据，即是否使用SGD
        :return: batches，元组list，0是特征数据，1是分类真实值
        """
        batches = []
        print(x_train.shape)
        print(y_train.shape)
        temp = np.concatenate((x_train, y_train),axis=1)
        if is_shuffle:
            np.random.shuffle(temp)
        temp_x = temp[..., :x_train.shape[1]]
        temp_y = temp[..., x_train.shape[1]:]
        for i in range(np.int(self.sample_num / self.batch_size)):
            x_batch = []
            y_batch = []
            for j in range(self.batch_size):
                # print(i * self.batch_size + j)
                x_batch.append(temp_x[i * self.batch_size + j])
                y_batch.append(temp_y[i * self.batch_size + j])
            batch = (np.array(x_batch), np.array(y_batch))
            batches.append(batch)
        return batches

    def logistic(self, x):
        """
        logistic模型计算分类概率h_j(x)
        :param x: 特征数据，dim1是样本，dim2是特征
        :return: 分类概率，dim1是样本，dim2是类别
        """
        scores = np.dot(x, self.theta.T)
        exp = np.exp(-scores)
        h_j = 1 / (1 + exp)
        return h_j

    def loss(self, x_batch, y_batch):
        """
        计算交叉熵损失
        :param x_batch: 特征数据，dim1是样本，dim2是特征
        :param y_batch: 分类真实值，one_hot，dim1是样本，dim2是类别
        :return:
        """
        p_c = self.logistic(x_batch)
        loss = - (1 / self.sample_num) * np.sum(y_batch * np.log(p_c))
        # loss = - np.sum(y_batch * np.log(p_c))
        return loss

    def gradient(self, x_batch, y_batch, h_j):
        """
        误差反向传播法（解析法）计算梯度
        :param x_batch: 特征数据，dim1是样本，dim2是特征
        :param y_batch: 分类真实值，one_hot，dim1是样本，dim2是类别
        :param h_j: 分类概率，dim1是样本，dim2是类别
        :return: 梯度值，dim1是类别，dim2是特征
        """
        grad = (1 / self.sample_num) * np.dot(x_batch.T, (h_j - y_batch))
        # grad = np.dot(x_batch.T, (h_j - y_batch))
        return grad
