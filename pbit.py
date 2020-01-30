import random
import numpy as np
import cupy as cp


class pcircuit:
    def __init__(self, Nm=0, J=[[]], h=[], beta=1, model="cpsl", delta_t=0.01):
        self.Nm = Nm
        self.J = np.array(J)
        self.h = np.array(h)
        self.beta = beta
        self.model = model
        self.dt = delta_t
        self.m = np.sign(np.add(np.random.rand(Nm) * 2, -1))

    def __call__(self):
        return "I am a p-circuit with " + str(self.Nm) + " p-bits."

    def setWeights(self, J, h):
        self.J = J
        self.h = h

    def setSize(self, num_pbits):
        self.Nm = num_pbits

    def setBeta(self, beta):
        self.beta = beta

    def setModel(self, model, dt=None):
        if dt is None:
            dt = self.dt
        self.model = model
        self.dt = dt

    def setState(self, m):
        self.m = m

    def getState(self):
        return self.m

    def getWeights(self):
        return self.J, self.h

    def getmodel(self):
        return self.model

    def getSize(self):
        return self.Nm

    def getBeta(self):
        return self.beta

    def reset(self):
        self.m = np.sign(np.add(np.random.rand(self.Nm) * 2, -1))

    def getBoltzmann(self):
        '''
        Retrieve the boltzmann statistics for a given p-circuit. Returns a normalized-to-1 np array of length 2^Nm with
        the exact probability of occupation for each state. Note, for Nm > 15, it may take a very long time for
        getBoltzmann() to execute. It must calculate a probability for each of 2^Nm states.
        '''

        all_state = [0 for i in range(2**self.Nm)]
        for cc in range(2**self.Nm):
            b = [int(x) for x in bin(cc)[2:]]
            state = [0] * (self.Nm - len(b))
            state.extend(b)
            state = np.array(state)  # convert to nd.array
            state = np.subtract(2*state, 1)  # make [-1,1] from [0,1]
            E = (np.dot(state, self.h) + np.multiply(0.5, np.dot(np.dot(state, self.J), state)))
            all_state[cc] = all_state[cc] + np.exp(-1*E)
        all_state = np.divide(all_state, np.sum(all_state))
        return np.array(all_state)

    def buildRandomNetwork(self, Nm, seed=0, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5):
        """
        build a random p-circuit. By default, only a random J will be made. If random_h is set to True, then 
        a random h will also be set with maximum values set by h_max_weight. 
        weight_type can be float, int
        """
        random.seed(seed)
        if weight_type == "float":
            self.Nm = Nm
            self.m = np.sign(np.add(np.random.rand(Nm) * 2, -1))
            self.J = (np.random.rand(Nm, Nm)*2-1) * J_max_weight/2
            self.J = self.J + np.transpose(self.J)
            np.fill_diagonal(self.J, 0)
            if random_h:
                self.h = np.add(np.random.rand(Nm) * 2, -1) * h_max_weight
            else:
                self.h = np.zeros((Nm))

    def runFor(self, Nt, model=None, gpu=False):
        """
        can provide an update scheme(cpsl, ppsl) to take effect for num_steps (Nt) timesteps, otherwise, the currently set 
        update scheme will take effect. If no update scheme has been selected, it will default to classical psl. Setting
        gpu to True will use a compatible CUDA enabled GPU if available.
        """
        if model is None:
            model = self.model
        if model == "cpsl":
            if gpu == False:
                m_all, self.m = cpsl(self.m, self.Nm, self.J, self.h, self.beta, Nt)
                return m_all
            else:
                m_all, self.m = cpsl_gpu(self.m, self.Nm, self.J, self.h, self.beta, Nt)
                return m_all
        elif model == "ppsl":
            if gpu == False:
                m_all, self.m = ppsl(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.dt)
                return m_all
            else:
                m_all, self.m = ppsl_gpu(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.dt)
                return m_all
        else:
            print("Error: unknown model")


def convertToBase10(a, inputBase=2):
    arr = np.array(a)
    Look = inputBase**np.arange(len(arr))
    return int(round(np.dot(a, Look)))


def cpsl(m, Nm, J, h, beta, Nt):
    m_all = np.zeros((Nt, Nm))
    for j in range(Nt):
        for i in np.random.permutation(Nm):
            xx = -1*beta * (np.dot(m, J[:, i]) + h[i])
            m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))
        m_all[j, :] = m
    m_all = np.array(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


def cpsl_gpu(m, Nm, J, h, beta, Nt):
    m_all = cp.zeros((Nt, Nm))
    m = cp.asarray(m)
    J = cp.asarray(J)
    h = cp.asarray(h)
    for j in range(Nt):
        for i in range(Nm):  # np.random.permutation(Nm):
           xx = beta * (cp.dot(m, J[:, i]) + h[i])
           m[i] = cp.sign(random.uniform(-1, 1) - cp.tanh(xx))
        m_all[j] = m
    m_all = cp.asnumpy(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


def ppsl(m, Nm, J, h, beta, Nt, dt):
    J = np.array(J)
    h = np.array(h)
    m_all = np.zeros((Nt, Nm))
    for i in range(Nt):
        x = np.multiply(np.add(np.dot(J, m), h), -1*beta)
        p = np.exp(-1*dt * np.exp(np.multiply(-1*m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(Nm))))
        m_all[i] = m
    m_all = np.array(m_all)
    m_all[m_all < 0] = 0
    return m_all, m

def ppsl_gpu(m, Nm, J, h, beta, Nt, dt):
    m = cp.asarray(m)
    J = cp.asarray(J)
    h = cp.asarray(h)
    m_all = cp.zeros((Nt, Nm))
    for i in range(Nt):
        x = cp.multiply(cp.add(cp.dot(J, m), h), -1*beta)
        p = cp.exp(-1*dt * cp.exp(cp.multiply(-1*m, x)))
        m = cp.multiply(m, cp.sign(cp.subtract(p, cp.random.rand(Nm))))
        m_all[i] = m
    m_all = cp.asnumpy(m_all)
    m_all[m_all < 0] = 0
    return m_all, m
