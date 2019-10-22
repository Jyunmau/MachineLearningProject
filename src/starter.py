import data_process as dp
import softmax as sm
import logistic as ls
import numpy as np

if __name__ == "__main__":
    sd = dp.SolveData()
    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    # sr = sm.SoftmaxRegression(0.1, 10, 10000)
    # sr.train(x1, y1, y2, True)
    lr = ls.LogisticRegression(1, 1, 500)
    temp_y2 = []
    for i in y2:
        temp = []
        temp.append(i)
        temp = np.array(temp)
        temp_y2.append(temp)
    temp_y2 = np.array(temp_y2)
    lr.train(x1, temp_y2, temp_y2, True)
