import numpy as np
import pbit

def convertToBase10(a, inputBase=2):
    try:
        b = a[0]
    except:
        print("Empty array sent to convertToBase10 function")
        return
    arr = np.flip(np.array(a), axis=0)
    length = len(arr[0]) if arr.ndim==2 else len(arr)
    Look = inputBase ** np.arange(length)
    return np.round(np.dot(arr, Look)).astype("int")


J = [[0, -0.4407, 0, 0, 0, 0],
     [-0.4407, 0, -0.4407, 0, 0, 0],
     [0, -0.4407, 0, 0, 0, 0],
     [0, 0, 0, 0, -0.4407, 0],
     [0, 0, 0, -0.4407, 0, -0.4407],
     [0, 0, 0, 0, -0.4407, 0]]
h = [0, 0, 0, 0, 0, 0]

mypcircuit = pbit.pcircuit(J=J, h=h)
m = mypcircuit.runFor(10)
decimal = convertToBase10(m)
print(decimal)
