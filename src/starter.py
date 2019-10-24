import data_process as dp
import softmax as sm
import logistic as ls
import numpy as np


#
# 该文件是执行调用示例
#

def main():
    # logistic regression
    sd = dp.SolveData()
    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    # GD
    lr_gd = ls.LogisticRegression(1, 1, 250)
    temp_y2 = sd.solve_logistic(y2)
    lr_gd.train(x1, temp_y2, temp_y2, True)
    # SGD
    lr_sgd = ls.LogisticRegression(1, 1, 250)
    lr_sgd.train(x1, temp_y2, temp_y2, False)

    # softmax regression
    # data1
    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    # GD
    sr_d1_gd = sm.SoftmaxRegression(1, 10, 250)
    sr_d1_gd.train(x1, y1, y2, False)
    # SGD
    sr_d1_sgd = sm.SoftmaxRegression(1, 10, 250)
    sr_d1_sgd.train(x1, y1, y2, True)
    # data2
    x1, y1 = sd.solve_2()
    x2, y2 = sd.solve_2(False)
    # GD
    sr_d2_gd = sm.SoftmaxRegression(10, 10, 150)
    sr_d2_gd.train(x1, y1, y2, False)
    # SGD
    sr_d2_sgd = sm.SoftmaxRegression(10, 10, 3)
    sr_d2_sgd.train(x1, y1, y2, True)


if __name__ == "__main__":
    main()
