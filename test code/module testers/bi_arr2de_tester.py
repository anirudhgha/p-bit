import numpy as np
import pbit
import matplotlib.pyplot as plt

# def bi_arr2de(a, inputBase=2):
#     try:
#         b = a[0]
#     except:
#         print("Empty array sent to convertToBase10 function")
#         return
#     arr = np.flip(np.array(a))
#     length = len(arr[0]) if arr.ndim == 2 else len(arr)
#     Look = inputBase ** np.arange(length)
#     return np.round(np.dot(arr, Look)).astype("int")

Nt = 100000

mypcircuit = pbit.pcircuit()
mypcircuit.load('and')
Nm = mypcircuit.getSize()
m = mypcircuit.generate_samples(Nt)

# run an array through convertToBase10
deci = pbit.bi_arr2de(m)
hist_multi = np.array([0 for i in range(2**Nm)])
for i in range(Nt):
    hist_multi[deci[i]] += 1

# run a single value through convertToBase10
hist_single = np.array([0 for i in range(2 ** Nm)])
for i in range(Nt):
    hist_single[pbit.bi_arr2de(m[i, :])] += 1            # build histogram of Nt states

# plot
barWidth = 0.25
x1 = np.arange(2 ** Nm)
x2 = np.array([x + barWidth for x in x1])

plt.bar(x1, hist_multi,  width=barWidth, edgecolor='white', label='Full array passed into convert...')
plt.bar(x2, hist_single,  width=barWidth, edgecolor='white', label="single value passed into convert...")
plt.legend()
plt.show()


