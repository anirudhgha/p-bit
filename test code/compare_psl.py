import numpy as np
import matplotlib.pyplot as plt
import random

I0 = 1
d_t = 1/15
Nt, Nm = 20000, 15  # Nm <= b size limit yields boltzmann curve as well
boltz_size_lim = 20

J = np.add(np.random.rand(Nm, Nm) * 2, -1)  # matrix of values between [-1, 1)
J = (J + J.T)/2
np.fill_diagonal(J, 0)
H = np.zeros(Nm)  # h is zero vector
# array of random ints either -1, 1
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))

energy_old = np.zeros(Nt * Nm)
energy_new = np.zeros(Nt * Nm)
if Nm <= boltz_size_lim:
    energy_boltz = np.zeros(pow(2, Nm))
print('total runs per simulation', Nt*Nm)


def boltzmann(J, H, Nt, Nm, energy_boltz):
    print('Finding boltzmann level...')
    for cc in range(pow(2, Nm)):
        b = [int(x) for x in bin(cc)[2:]]
        state = [0] * (Nm - len(b))
        # iz = jj converted to binary and padded with zeros make length Nm
        state.extend(b)
        state = np.array(state)  # convert to nd.array
        state = np.subtract(2*state, 1)  # make [-1,1] from [0,1]
        energy_boltz[cc] = I0 * (np.dot(state, H) + np.multiply(0.5, np.dot(np.dot(state, J), state)))
    if np.mod(cc, (int(pow(2, Nm) / 5))) == 0:
        print(pow(2, Nm) - cc)
    energy_boltz = np.subtract(energy_boltz, np.sum(energy_boltz))
    eb = np.sum(np.multiply(energy_boltz, np.exp(-1*energy_boltz))) / \
        np.sum(np.exp(-1*energy_boltz))
    return np.multiply(eb, np.ones(Nt * Nm))


def old_psl(m, J, H, energy_old):
    cc = 0  # measure time using clock cycles instead of program execution time
    print('Finding curve for old psl...')
    for j in range(Nt):
        for i in np.random.permutation(Nm):
            I = I0 * (np.dot(m, J[:, i]) + H[i])
            m[i] = np.sign(random.uniform(-1, 1) - np.tanh(I))
            energy_old[cc] = I0 * (np.dot(m, H) +
                                   np.multiply(0.5, np.dot(np.dot(m, J), m)))
            cc += 1
            if np.mod(cc, (Nt*Nm)/5) == 0:
                print(Nm*Nt-cc)
    energy_old = np.divide(np.cumsum(energy_old), np.arange(1, Nt * Nm + 1))
    return energy_old


def new_psl(m, J, H, dt, energy_new):
    cc = 0
    print('Finding curve for new psl...')
    for i in range(Nt * Nm):
        x = np.multiply(np.add(np.dot(J, m), H), -1*I0)
        p = np.exp(-1*dt * np.exp(np.multiply(-1*m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(Nm))))
        energy_new[cc] = I0 * \
            (np.dot(m, H) + np.multiply(0.5, np.dot(np.dot(m, J), m)))
        cc += 1
        if np.mod(cc, (Nt*Nm)/5) == 0:
            print(Nm*Nt-cc)
    energy_new = np.divide(np.cumsum(energy_new), np.arange(1, Nt*Nm+1))
    return energy_new


# run functions
energy_old = old_psl(m, J, H, energy_old)
m = np.sign(np.add(np.random.rand(Nm) * 2, -1))  # reset m
energy_new = new_psl(m, J, H, d_t, energy_new)
if Nm <= boltz_size_lim:
    energy_boltz = boltzmann(J, H, Nt, Nm, energy_boltz)

# plot the energies
clocks = np.arange(Nm * Nt)
lim = pow(2, Nm)
plt.plot(clocks, energy_old[0:Nm*Nt], label='Classical-PSL')
plt.plot(clocks, energy_new[0:Nm*Nt], label='Parallel-PSL')
if Nm <= boltz_size_lim:
    plt.plot(clocks, energy_boltz[0:Nm*Nt], label='Boltzmann')
plt.title('CPSL vs PPSL(dt = {}), size = {}'.format(round(d_t, 4), Nm))
plt.xlabel('Clock cycles')
plt.ylabel('Energy')
plt.legend()
plt.show(block=True)
