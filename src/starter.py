import data_process as dp
import softmax as sm

if __name__ == "__main__":
    sd = dp.SolveData()
    x1, y1 = sd.solve_1()
    sr = sm.SoftmaxRegression(0.1, 4, 1000)
    sr.train(x1, y1)
