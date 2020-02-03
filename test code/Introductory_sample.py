"""
Simple sample code running an and gate p-circuit
"""
import pbit
import matplotlib.pyplot as plt

# build p-circuit
J = [[0, -2, -2],
     [-2, 0, 1],
     [-2, 1, 0]]
h = [2, -1, -1]
myp = pbit.pcircuit(J, h)

# run p-circuit
myp.draw()
m_list = myp.runFor(1000000, gpu=True)

# plot
m_list = pbit.convertToBase10(m_list)
plt.hist(m_list)
plt.show()
