import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def xrange(x):
    n = 0
    while n < x:
        yield n
        n += 1


def main():
    numframes = 10
    numpoints = 10
    color_data = np.ones((numframes, numpoints))
    x, y, c = np.random.random((3, numpoints))
    area_x = np.arange(0, 1, 0.2)
    area_y = np.arange(0, 1, 0.2)
    area = []
    for i in range(5):
        for j in range(5):
            temp_x = []
            temp_x.append(area_x[i])
            temp_x.append(area_y[j])
            temp_x = np.array(temp_x)
            area.append(temp_x)
    area = np.array(area)
    x1 = area[..., :1]
    y1 = area[..., 1:]

    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=100)
    plt.scatter(x1, y1, c='r', s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  fargs=(color_data, scat))
    plt.show()


def update_plot(i, data, scat):
    scat.set_offsets(np.random.random((10, 2)))
    return scat,


main()
