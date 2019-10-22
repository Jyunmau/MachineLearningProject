import data_process as dp
import softmax as sm
import logistic as ls

if __name__ == "__main__":
    sd = dp.SolveData()
    x1, y1 = sd.solve_1()
    x2, y2 = sd.solve_1(False)
    sr = sm.SoftmaxRegression(1, 1, 10000)
    sr.train(x1, y1, y2)
    # lr = ls.LogisticRegression(1, 1, 500)
    # lr.train(x1, y1)
