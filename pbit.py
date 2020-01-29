import random
import numpy as np


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

    def getBoltzmann(self, Nm=None):
        if Nm is None:
            Nm = self.Nm
            '''
            fill in code for boltzmann
            '''

    def buildRandomNetwork(self, Nm, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5):
        """
        build a random p-circuit. By default, only a random J will be made. If random_h is set to True, then 
        a random h will also be set with maximum values set by h_max_weight. 
        weight_type can be float, int
        """
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

    def saveSteps(self, Nt, model=None):
        """
        can provide an update scheme(cpsl, ppsl) to take effect for num_steps (Nt) timesteps, otherwise, the currently set 
        update scheme will take effect. If no update scheme has been selected, it will default to classical psl. 
        """
        if model is None:
            model = self.model
        if model == "cpsl":
            return self.cpsl(self.Nm, self.J, self.h, self.beta, Nt)
        elif model == "ppsl":
            return self.ppsl(self.Nm, self.J, self.h, self.beta, Nt, self.dt)
        else:
            print("Error: unknown model")

    def cpsl(self, Nm, J, h, beta, Nt):
        J = np.array(J)
        h = np.array(h)
        m_all = np.zeros((Nt,Nm))
        for j in range(Nt):
            for i in range(Nm): #np.random.permutation(Nm):
                xx = beta * (np.dot(self.m, J[:, i]) + h[i])
                self.m[i] = np.sign(random.uniform(-1, 1) - np.tanh(xx))
            m_all[j] = self.m
        m_all = np.array(m_all)
        m_all[m_all < 0] = 0
        return m_all

    def ppsl(self, Nm, J, h, beta, Nt, dt):
        J = np.array(J)
        h = np.array(h)
        m_all = np.zeros((Nt,Nm))
        for i in range(Nt):
            x = np.multiply(np.add(np.dot(J, self.m), h), -1*beta)
            p = np.exp(-1*dt * np.exp(np.multiply(-1*self.m, x)))
            self.m = np.multiply(self.m, np.sign(np.subtract(p, np.random.rand(Nm))))
            m_all[i] = self.m
        m_all = np.array(m_all)
        m_all[m_all < 0] = 0
        return np.array(m_all)
