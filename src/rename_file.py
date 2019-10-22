# python批量更换后缀名
import os

# 列出当前目录下所有的文件

path_1 = "data/ex4Data/"
path_2 = "data/Machine Learning Data/data/"
files = os.listdir(path_2)
print('files', files)
for filename in files:
    portion = os.path.splitext(filename)
    # 如果后缀是.dat
    if portion[1] == ".dat":
        # 重新组合文件名和后缀名

        newname = path_2 + portion[0] + ".txt"
        os.rename(path_2 + filename, newname)
