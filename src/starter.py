import data_process as dp
import softmax as sm
import logistic as ls
import numpy as np

if __name__ == "__main__":
    sd = dp.SolveData()
    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    # lr_gd = ls.LogisticRegression(1, 1, 250)
    # temp_y2 = []
    # for i in y2:
    #     temp = []
    #     temp.append(i)
    #     temp = np.array(temp)
    #     temp_y2.append(temp)
    # temp_y2 = np.array(temp_y2)
    # lr_gd.train(x1, temp_y2, temp_y2, True)
    # lr_sgd = ls.LogisticRegression(1, 1, 250)
    # temp_y2 = []
    # for i in y2:
    #     temp = []
    #     temp.append(i)
    #     temp = np.array(temp)
    #     temp_y2.append(temp)
    # temp_y2 = np.array(temp_y2)
    # lr_sgd.train(x1, temp_y2, temp_y2, False)

    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    sr_d1_gd = sm.SoftmaxRegression(1, 10, 250)
    sr_d1_gd.train(x1, y1, y2, False)
    sr_d1_sgd = sm.SoftmaxRegression(1, 10, 250)
    sr_d1_sgd.train(x1, y1, y2, True)
    x1, y1 = sd.solve_2()
    x2, y2 = sd.solve_2(False)
    sr_d2_gd = sm.SoftmaxRegression(10, 10, 150)
    sr_d2_gd.train(x1, y1, y2, False)
    # sr_d2_sgd = sm.SoftmaxRegression(10, 10, 3)
    # sr_d2_sgd.train(x1, y1, y2, True)
