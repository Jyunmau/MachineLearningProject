import numpy as np
from matplotlib import pyplot as plt


# _*_ coding:utf-8 _*_

#   @Version : 0.2.0
#   @Time    : 2019/10/22 23:24
#   @Author  : Jyunmau Chan
#   @File    : data_process.py


class SolveData:
    """处理模型数据"""

    def __init__(self):
        self.path_1 = "data/ex4Data/"
        self.path_2 = "data/Machine Learning Data/data/"

    def __one_hot(self, y, category_num):
        """
        将类别编码转为one-hot编码
        :param y: 分类真实值，dim1是样本，值是类别
        :return: 分类真实值，one_hot，dim1是样本，dim2是类别
        """
        # category_num = 2
        sample_num = y.shape[0]
        one_hot = np.zeros((sample_num, category_num)).astype('int64')
        one_hot[np.arange(sample_num).astype('int64'), y.astype('int64').T] = 1
        return one_hot

    def __normalize(self, data_array):
        """
        最大最小值归一化
        :param data_array: 特征数据，dim1是样本，dim2是特征
        :return: 归一化后的特征数据，dim1是样本，dim2是特征
        """
        dot_x1 = []
        dot_x2 = []
        for i in data_array:
            dot_x1.append(i[0])
            dot_x2.append(i[1])
        max_min_x1 = (np.max(dot_x1), np.min(dot_x1))
        max_min_x2 = (np.max(dot_x2), np.min(dot_x2))
        dot_x1_new = []
        dot_x2_new = []
        for i in dot_x1:
            dot_x1_new.append((i - max_min_x1[1]) / (max_min_x1[0] - max_min_x1[1]))
        for i in dot_x2:
            dot_x2_new.append((i - max_min_x2[1]) / (max_min_x2[0] - max_min_x2[1]))
        data_array_new = []
        for i in range(data_array.shape[0]):
            data_array_new.append([dot_x1_new[i], dot_x2_new[i]])
        data_array_new = np.array(data_array_new)
        return data_array_new

    def solve_1(self, is_one_hot=True):
        """
        从文件中读取数据并处理格式
        :return: 特征数据，dim1是样本，dim2是特征
                  分类真实值，one_hot，dim1是样本，dim2是类别
        """
        x = np.loadtxt(self.path_1 + 'ex4x.txt')
        x = self.__normalize(x)
        y = np.loadtxt(self.path_1 + 'ex4y.txt')
        if is_one_hot:
            y = self.__one_hot(y, 2)
        # 添加偏置项
        sample_num, feature_num = x.shape
        bias_x = np.ones((sample_num, 1))
        x = np.hstack((x, bias_x))
        return x, y

    def solve_2(self, is_one_hot=True):
        x = np.loadtxt(self.path_2 + 'iris_x.txt')
        x = self.__normalize(x)
        y = np.loadtxt(self.path_2 + 'iris_y.txt')
        if is_one_hot:
            y = self.__one_hot(y, 3)
        # 添加偏置项
        sample_num, feature_num = x.shape
        bias_x = np.ones((sample_num, 1))
        x = np.hstack((x, bias_x))
        return x, y


class PlotData:
    """对模型数据作图"""

    def __solve_data(self, data_array):
        """
        将特征数据按特征维度分开
        :param data_array: 特征数据，dim1是样本，dim2是特征
        :return: 每个特征维度输出一个list，dim是样本
        """
        dot_x = []
        dot_y = []
        for i in data_array:
            dot_x.append(i[0])
            dot_y.append(i[1])
        return dot_x, dot_y

    def plot_data_1(self, data_array, category_array):
        """
        输出样本数据散点图
        :param data_array: 特征数据，dim1是样本，dim2是特征
        :param category_array: 分类真实值，dim1是样本，值是类别
        :return:
        """
        data_1 = []
        data_2 = []
        for i in range(80):
            if category_array[i] == 0:
                data_1.append(data_array[i])
            elif category_array[i] == 1:
                data_2.append(data_array[i])
        dot_x, dot_y = self.__solve_data(data_1)
        plt.plot(dot_x, dot_y, 'ob')
        dot_x, dot_y = self.__solve_data(data_2)
        plt.plot(dot_x, dot_y, 'or')
        plt.show()

    def plot_data_2(self, data_array, category_array):
        """
        输出样本数据散点图
        :param data_array: 特征数据，dim1是样本，dim2是特征
        :param category_array: 分类真实值，dim1是样本，值是类别
        :return:
        """
        data_1 = []
        data_2 = []
        data_3 = []
        for i in range(len(category_array)):
            if category_array[i] == 0:
                data_1.append(data_array[i])
            elif category_array[i] == 1:
                data_2.append(data_array[i])
            elif category_array[i] == 2:
                data_3.append(data_array[i])
        dot_x, dot_y = self.__solve_data(data_1)
        plt.plot(dot_x, dot_y, 'ob')
        dot_x, dot_y = self.__solve_data(data_2)
        plt.plot(dot_x, dot_y, 'or')
        dot_x, dot_y = self.__solve_data(data_3)
        plt.plot(dot_x, dot_y, 'oy')
        plt.show()

    def plot_loss(self, losses):
        """
        打印loss值，横轴步骤，纵轴是loss
        :param losses: 损失的list，dim1是步骤
        :return:
        """
        loss_num = len(losses)
        step_axis = np.arange(0, loss_num)
        fig, ax = plt.subplots()
        # # ax.scatter(step_axis, losses, c='b', marker='.')
        # # plt.show()
        # plt.ion()
        # # for i in range(loss_num):
        # #     ax.scatter(step_axis[i], losses[i], c='b', marker='.')
        # #     plt.pause(0.001)
        # # plt.ioff()
        # for i in range(loss_num):
        #     temp_losses = losses[:i]
        #     temp_step_axis = step_axis[:i]
        #     plt.cla()
        #     ax.scatter(temp_step_axis, temp_losses, c='b', marker='.')
        #     plt.pause(0.001)
        # plt.ioff()
        # plt.show()
        plt.plot(step_axis, losses, '.r')
        plt.show()

    def plot_result(self, data_array, category_array, theta_list):
        plt.ion()
        for theta_mat in theta_list:
            plt.cla()
            area_x = np.arange(0, 1, 0.02)
            area_y = np.arange(0, 1, 0.02)
            area = []
            for i in range(50):
                for j in range(50):
                    temp_x = []
                    temp_x.append(area_x[i])
                    temp_x.append(area_y[j])
                    temp_x.append(1)
                    temp_x = np.array(temp_x)
                    area.append(temp_x)
            area = np.array(area)
            scores = np.dot(area, theta_mat.T)
            exp = np.exp(scores)
            sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
            h_j = exp / sum_exp
            for i in range(area.shape[0]):
                if np.argmax(h_j[i]) == 0:
                    plt.plot(area[i][0], area[i][1], color='lightpink', marker='o')
                elif np.argmax(h_j[i]) == 1:
                    plt.plot(area[i][0], area[i][1], color='lightgreen', marker='o')
                elif np.argmax(h_j[i]) == 2:
                    plt.plot(area[i][0], area[i][1], color='skyblue', marker='o')
            data_array = data_array[..., :2]
            data_1 = []
            data_2 = []
            data_3 = []
            for i in range(len(category_array)):
                if category_array[i] == 0:
                    data_1.append(data_array[i])
                elif category_array[i] == 1:
                    data_2.append(data_array[i])
                elif category_array[i] == 2:
                    data_3.append(data_array[i])
            dot_x, dot_y = self.__solve_data(data_1)
            plt.plot(dot_x, dot_y, 'ob')
            dot_x, dot_y = self.__solve_data(data_2)
            plt.plot(dot_x, dot_y, 'or')
            dot_x, dot_y = self.__solve_data(data_3)
            plt.plot(dot_x, dot_y, 'oy')
            plt.pause(0.001)
            # temp_x1 = np.linspace(0, 1)
            # if theta_mat.shape[0] == 1 and theta_mat.shape[1] == 3:
            #     temp_x2 = - (theta_mat[0][2] + temp_x1 * theta_mat[0][0]) / theta_mat[0][1]
            #     plt.plot(temp_x1, temp_x2)
            # else:
            #     for i in range(theta_mat.shape[0]):
            #         if theta_mat.shape[0] == 3 and i == 1:
            #             continue
            #         if theta_mat.shape[0] == 2 and i == 1:
            #             continue
            #         temp_x2 = (theta_mat[i][2] - theta_mat[1][2] + temp_x1
            #                    * (theta_mat[i][0] - theta_mat[1][0])) / (theta_mat[1][1] - theta_mat[i][1])
            #         plt.plot(temp_x1, temp_x2)
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    sd = SolveData()
    PL = PlotData()
    x1, y1 = sd.solve_1(False)
    PL.plot_data_2(x1, y1)
