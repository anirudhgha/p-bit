import random
import numpy as np
import cupy as cp
import turtle
import math


class pcircuit:
    def __init__(self, J=[[]], h=[], beta=1, Nm=0, model="cpsl", delta_t=0.01, start_beta=1, end_beta=2,
                 growth_factor=1.001, anneal="constant"):
        self.J = np.array(J)
        self.h = np.array(h)
        if self.J is None:
            self.Nm = 0
        else:
            self.Nm = len(self.J)
        self.model = model
        self.dt = delta_t
        self.m = np.sign(np.add(np.random.rand(self.Nm) * 2, -1))
        # annealing
        self.beta = beta
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.growth_factor = growth_factor
        self.beta = beta
        self.anneal = anneal

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

    def setAnneal(self, anneal="constant", beta=1, start_beta=0, end_beta=1, growth_factor=1):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.growth_factor = growth_factor
        self.beta = beta
        self.anneal = anneal

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
        """
         Retrieve the boltzmann statistics for a given p-circuit. Returns a normalized-to-1 np array of length 2^Nm with
        the exact probability of occupation for each state. Note, for Nm > 15, it may take a very long time for
        getBoltzmann() to execute. It must calculate a probability for each of 2^Nm states.
        :return:
        """

        all_state = [0 for i in range(2 ** self.Nm)]
        for cc in range(2 ** self.Nm):
            b = [int(x) for x in bin(cc)[2:]]
            state = [0] * (self.Nm - len(b))
            state.extend(b)
            state = np.array(state)  # convert to nd.array
            state = np.subtract(2 * state, 1)  # make [-1,1] from [0,1]
            E = (np.dot(state, self.h) + np.multiply(0.5, np.dot(np.dot(state, self.J), state)))
            all_state[cc] = all_state[cc] + np.exp(-1 * E)
        all_state = np.divide(all_state, np.sum(all_state))
        return np.array(all_state)

    def buildRandomNetwork(self, Nm, seed=0, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5):
        """
        build a random p-circuit. By default, only a random J will be made. If random_h is set to True, then
        a random h will also be set with maximum values set by h_max_weight.
        weight_type can be float, int.
        :param Nm:
        :param seed:
        :param weight_type:
        :param J_max_weight:
        :param random_h:
        :param h_max_weight:
        :return:
        """
        random.seed(seed)
        if weight_type == "float":
            self.Nm = Nm
            self.m = np.sign(np.add(np.random.rand(Nm) * 2, -1))
            self.J = (np.random.rand(Nm, Nm) * 2 - 1) * J_max_weight / 2
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

        Annealing options: constant (beta), linear (beta_start, beta_end),
        geometric (beta_start, beta_end, growth_factor)
        :param Nt:
        :param model:
        :param gpu:
        :return:
        """

        annealing_factors = [self.beta, self.start_beta, self.end_beta, self.growth_factor]

        if model is None:
            model = self.model
        if model == "cpsl":
            if not gpu:
                m_all, self.m = _cpsl(self.m, self.Nm, self.J, self.h, Nt, self.anneal, annealing_factors)
                return m_all
            else:
                m_all, self.m = _cpsl_gpu(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.anneal,
                                          annealing_factors)
                return m_all
        elif model == "ppsl":
            if not gpu:
                m_all, self.m = _ppsl(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.dt, self.anneal,
                                      annealing_factors)
                return m_all
            else:
                m_all, self.m = _ppsl_gpu(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.dt, self.anneal,
                                          annealing_factors)
                return m_all
        else:
            print("Error: unknown model")

    def draw(self, labels=True):
        """
        Draw out your pcircuit!
        :param labels:
        :return:
        """
        if self.Nm == 0:
            print("Cannot draw 0 Nm pcircuit. Use setSize() to pass a network size, "
                  "or re-construct the pcircuit with pbit.Pcircuit()")
            return

        # setup
        # resize screen to fit
        # Nm*100 == 2*pi*r
        # Nm*100/(2*pi) == r
        # screensize = 2r+200 = (Nm*100/pi)+200
        turtle.screensize((self.Nm * 100 / np.pi) + 200, (self.Nm * 100 / np.pi) + 400)

        turtle.speed("fastest")
        turtle.hideturtle()
        turtle.penup()
        turtle.setpos(0, 200)
        turtle.pendown()
        turn = 360 / self.Nm

        # place pbits
        pbitpos = np.zeros((self.Nm, 2))
        for i in range(self.Nm):
            pbitpos[i, 0], pbitpos[i, 1] = turtle.pos()
            turtle.dot(size=40)
            turtle.penup()
            turtle.right(turn)
            turtle.forward(100)

        # draw weights
        drawn = []
        for i in range(self.Nm):  # source pbit
            for j in range(i, self.Nm):  # destination pbit
                if self.J[i, j]:
                    turtle.penup()
                    turtle.goto(pbitpos[i, 0], pbitpos[i, 1])
                    turtle.pendown()
                    turtle.goto(pbitpos[j, 0], pbitpos[j, 1])
                    turtle.penup()

                    if labels:  # label weights
                        turtle.penup()
                        turtle.goto(pbitpos[i, 0], pbitpos[i, 1])
                        turtle.setheading(turtle.towards(pbitpos[j, 0], pbitpos[j, 1]))
                        turtle.forward(_dist(pbitpos[i], pbitpos[j]) / 2)
                        if (pbitpos[i, 1] > pbitpos[j, 1]):
                            turtle.setheading(turtle.towards(pbitpos[j, 0], pbitpos[j, 1]))
                            turtle.left(90)
                        else:
                            turtle.setheading(turtle.towards(pbitpos[j, 0], pbitpos[j, 1]))
                            turtle.left(90)
                        if np.abs(pbitpos[i, 1] - pbitpos[j, 1]) < 1:
                            turtle.right(90)
                        turtle.forward(10)
                        if _not_drawn(i, j, drawn):
                            turtle.pendown()
                            turtle.write(str("{:.2f}".format(self.J[i, j])), font=("Calibri", 13, "normal"))
                            turtle.penup()

                        drawn.append([i, j])
        # hold drawing until click
        turtle.exitonclick()


def convertToBase10(a, inputBase=2):
    try:
        b = a[0]
    except IndexError:
        print("Empty array sent to convertToBase10 function")
        return
    try:
        inputBase > 0
    except:
        print("Input base must be larger than 0")
        return
    arr = np.flip(np.array(a), axis=0)
    length = len(arr[0]) if arr.ndim == 2 else len(arr)
    Look = inputBase ** np.arange(length)
    return np.round(np.dot(arr, Look)).astype("int")


def errorMSE(predicted, exact):
    return np.sqrt(np.divide(np.sum(np.sum((np.abs(np.subtract(exact, predicted))) ** 2)),
                             np.sum(np.sum((np.abs(exact)) ** 2))))


def _incrementAnnealing(cur_beta, anneal, annealing_factors, Nt, start=False):
    if start:
        return annealing_factors[0] if anneal == "constant" else annealing_factors[1]
    if anneal == "constant":
        return cur_beta
    if anneal == "linear":
        return cur_beta + (annealing_factors[1] - annealing_factors[0]) / Nt
    if anneal == "geometric":
        return cur_beta * annealing_factors[2] if (cur_beta * annealing_factors[2] < annealing_factors[1]) else \
            annealing_factors[1]


def _cpsl(m, Nm, J, h, Nt, anneal, annealing_factors):
    beta = _incrementAnnealing(1, anneal, annealing_factors, Nt, start=True)
    m_all = np.zeros((Nt, Nm))
    for j in range(Nt):
        for i in np.random.permutation(Nm):
            xx = -1 * beta * (np.dot(m, J[:, i]) + h[i])
            m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))
            beta = _incrementAnnealing(beta, anneal, annealing_factors, Nt)
        m_all[j, :] = m
    m_all = np.array(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


def _cpsl_gpu(m, Nm, J, h, beta, Nt, anneal, annealing_factors):
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


def _ppsl(m, Nm, J, h, beta, Nt, dt, anneal, annealing_factors):
    J = np.array(J)
    h = np.array(h)
    m_all = np.zeros((Nt, Nm))
    for i in range(Nt):
        x = np.multiply(np.add(np.dot(J, m), h), -1 * beta)
        p = np.exp(-1 * dt * np.exp(np.multiply(-1 * m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(Nm))))
        m_all[i] = m
    m_all = np.array(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


def _ppsl_gpu(m, Nm, J, h, beta, Nt, dt, anneal, annealing_factors):
    m = cp.asarray(m)
    J = cp.asarray(J)
    h = cp.asarray(h)
    m_all = cp.zeros((Nt, Nm))
    for i in range(Nt):
        x = cp.multiply(cp.add(cp.dot(J, m), h), -1 * beta)
        p = cp.exp(-1 * dt * cp.exp(cp.multiply(-1 * m, x)))
        m = cp.multiply(m, cp.sign(cp.subtract(p, cp.random.rand(Nm))))
        m_all[i] = m
    m_all = cp.asnumpy(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


def _dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def _not_drawn(i, j, drawn):
    for ii in drawn:
        if math.isclose(i, ii[0]) and math.isclose(j, ii[1]):
            return False
    return True
