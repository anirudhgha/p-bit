import numpy as np


J = np.genfromtxt('5q3r.txt', delimiter=',')
print('[[')
for i in range(15):
    for k in range(15):
        print(J[i,k])
        if k < 14:
            print(',')

