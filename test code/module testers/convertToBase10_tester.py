import numpy as np
import pbit
import matplotlib.pyplot as plt

def convertToBase10(a, inputBase=2):
    try:
        b = a[0]
    except:
        print("Empty array sent to convertToBase10 function")
        return
    arr = np.flip(np.array(a))
    length = len(arr[0]) if arr.ndim == 2 else len(arr)
    Look = inputBase ** np.arange(length)
    return np.round(np.dot(arr, Look)).astype("int")


J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]
Nt = 100000
Nm=len(J)

mypcircuit = pbit.pcircuit(J=J, h=h)
m = mypcircuit.runFor(Nt)

# run an array through convertToBase10
deci = pbit.convertToBase10(m)
hist_multi = np.array([0 for i in range(2**Nm)])
for i in range(Nt):
    hist_multi[deci[i]] += 1

# run a single value through convertToBase10
hist_single = np.array([0 for i in range(2 ** Nm)])
for i in range(Nt):
    hist_single[pbit.convertToBase10(m[i, :])] += 1            # build histogram of Nt states

# plot
barWidth = 0.25
x1 = np.arange(2 ** Nm)
x2 = np.array([x + barWidth for x in x1])

plt.bar(x1, hist_multi,  width=barWidth, edgecolor='white', label='Full array passed into convert...')
plt.bar(x2, hist_single,  width=barWidth, edgecolor='white', label="single value passed into convert...")
plt.legend()
plt.show()


