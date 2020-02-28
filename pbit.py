import random
import numpy as np
import turtle
import math
from numba import jit, float64, int64
from numba import cuda

class pcircuit:
    def __init__(self, J=[[]], h=[], beta=1, Nm=0, model="cpsl", delta_t=None, start_beta=1, end_beta=2,
                 growth_factor=1.001, anneal="constant"):
        self.J = np.array(J)
        self.h = np.array(h)
        if self.J is None:
            self.Nm = 0
        else:
            self.Nm = len(self.J)
        self.model = model
        if delta_t is None:
            self.dt = 1 / (2 * self.Nm)
        else:
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
        self.J = np.array(J)
        self.h = np.array(h)

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

    def load_random(self, Nm, seed=0, weight_type="float", J_max_weight=5, random_h=False, h_max_weight=5):
        """
        load random weights into the p-circuit. By default, only a random J will be made. If random_h is set to True, then
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
        np.random.seed(seed)
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

    def generate_samples(self, Nt, model=None, gpu=False, ret_base ='binary'):
        """
        can provide an update scheme(cpsl, ppsl) to take effect for num_steps (Nt) timesteps, otherwise, the currently set
        update scheme will take effect. If no update scheme has been selected, it will default to classical psl. Setting
        gpu to True will use a compatible CUDA enabled GPU if available.

        Annealing options: constant (beta), linear (beta_start, beta_end),
        geometric (beta_start, beta_end, growth_factor)
        :param Nt: number of samples to generate (length of time to 'run' network if you collect 1 sample per time unit)
        :param model: cpsl or ppsl
        :param gpu: gpu currently does not support annealing, only a constant annealing scheme beta
        :return:
        """
        if Nt == 0:
            print('Error: Nt must be greater than 0')
            return
        Nt = int(Nt)
        annealing_factors = [self.beta, self.start_beta, self.end_beta, self.growth_factor]

        # gpu requires specific input types
        gpu_beta = float64(self.beta)
        gpu_m = float64(self.m)
        gpu_J = np.ndarray.flatten(np.ndarray.astype(self.J, "float64"))
        gpu_h = np.ndarray.flatten(np.ndarray.astype(self.h, "float64"))
        gpu_Nm = np.int64(self.Nm)
        gpu_rand = np.random.uniform(-1, 1, Nt * self.Nm)

        if model is None:
            model = self.model
        if model == "cpsl":
            if not gpu:
                m_all, self.m = _cpsl(self.m, self.Nm, self.J, self.h, Nt, self.anneal, annealing_factors)
            else:
                m_all = _cpsl_fast(gpu_m, gpu_Nm, gpu_J, gpu_h, gpu_beta, Nt, gpu_rand)
                m_all = np.reshape(m_all, (Nt, self.Nm))
                self.m = m_all[Nt - 1, :]
        elif model == "ppsl":
            if not gpu:
                m_all, self.m = _ppsl(self.m, self.Nm, self.J, self.h, self.beta, Nt, self.dt, self.anneal,
                                      annealing_factors)
            else:
                m_all = _ppsl_fast(gpu_m, gpu_Nm, gpu_J, gpu_h, gpu_beta, Nt, self.dt, gpu_rand)
                m_all = np.reshape(m_all, (Nt, self.Nm))
                self.m = m_all[Nt - 1, :]
        else:
            print("Error: unknown model")

        if ret_base == 'samples' or ret_base == 'sample' or ret_base == 'b' or ret_base == 'binary':
            return m_all
        elif ret_base == 'decimal' or ret_base == 'd' or ret_base == 'deci' or ret_base == 'decimals':
            return bi_arr2de(m_all)


    def draw(self, labels=True):
        """
        Draw out your pcircuit!
        :param labels: turn labels off to skip drawing out the weights, which slow down the draw
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
    def load_image_as_ground_state(self, file_name=None):
        import PIL.Image as Image
        if file_name is None:
            print("ERROR: no filename specified")
            return
        orig = Image.open('32x32.png')
        baw = orig.convert('1') #convert to black and white (baw)
        width, height = baw.size
        if width != height:
            print("ERROR: image must be a square")
            return
        J_temp = np.ones((height * width, height * width))

        # opposite pixels get weight of 1
        for row in range(height):
            for col in range(width):
                cur = row * width + col
                if col + 1 < width - 1:
                    right = row * width + (col + 1)
                else:
                    right = 0
                if col - 1 > 0:
                    left = row * width + (col - 1)
                else:
                    left = width - 1
                if row - 1 > 0:
                    up = (row - 1) * width + col
                else:
                    up = height - 1
                if row + 1 < height - 1:
                    down = (row + 1) * width + col
                else:
                    down = 0
                if col + 1 < width and baw.getpixel((col, row)) != baw.getpixel((col + 1, row)):
                    J_temp[cur, right] = -1
                if row + 1 < height and baw.getpixel((col, row)) != baw.getpixel((col, row + 1)):
                    J_temp[cur, down] = -1
                if col - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col - 1, row)):
                    J_temp[cur, left] = -1
                if row - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col, row - 1)):
                    J_temp[cur, up] = -1
        h = np.zeros(width*height)

        self.J = np.array(J_temp)
        self.h = np.array(h)
        self.Nm = np.int64(self.J.shape[0])
        self.reset() # sets up an initial m state

    def load(self, name=None):
        if name is None:
            print("ERROR: no data-name specified")
            return
        if name == "and":
            self.J = np.array([[0, -2, -2],
                               [-2, 0, 1],
                               [-2, 1, 0]])
            self.h = np.array([2, -1, -1])
            self.Nm = 3
            self.reset()
        elif name == "8q3r":
             self.J = np.genfromtxt('5q3r.txt', delimiter=',')
             self.h = np.zeros(24)
             self.Nm = 24
             self.reset()
        elif name == "2q3r":
            self.J = np.array([[0, -0.4407, 0, 0, 0, 0],
                               [-0.4407, 0, -0.4407, 0, 0, 0],
                               [0, -0.4407, 0, 0, 0, 0],
                               [0, 0, 0, 0, -0.4407, 0],
                               [0, 0, 0, -0.4407, 0, -0.4407],
                               [0, 0, 0, 0, -0.4407, 0]])
            self.h = np.zeros(6)
            self.Nm = 6
            self.reset()
        elif name == "not":
            self.J = np.array([[0,1],
                               [1,0]])
            self.h = np.zeros(2)
            self.Nm = 2
            self.reset()

def bi_arr2de(a, inputBase=2):
    try:
        b = a[0]
    except IndexError:
        print("Empty array sent to bi_arr2de function")
        return
    try:
        inputBase > 0
    except:
        print("Input base must be larger than 0")
        return
    try:
        b = a[0]
    except:
        print("Error: please send array to convert")
        return
    a[a < 0] = 0
    arr = np.flip(np.array(a))
    length = len(arr[0]) if arr.ndim == 2 else len(arr)
    Look = inputBase ** np.arange(length)
    return np.round(np.dot(arr, Look)).astype("int")


def errorMSE(predicted_arr, exact_arr):
    return np.sqrt(np.divide(np.sum(np.sum((np.abs(np.subtract(exact_arr, predicted_arr))) ** 2)),
                             np.sum(np.sum((np.abs(exact_arr)) ** 2))))

def live_heatmap(m_all, num_samples_to_plot=None, hold_time=0.5):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from functools import reduce

    Nt = m_all.shape[0]  # num rows in m_all is num samples taken
    Nm = m_all.shape[1]

    #find the factors of n to get the most "square" display we can get
    factors = list(reduce(list.__add__,
                          ([i, Nm // i] for i in range(1, int(Nm ** 0.5) + 1) if Nm % i == 0)))
    width = factors[-2]
    height = factors[-1]

    if num_samples_to_plot is None:
        num_samples_to_plot = Nm
    m_live = m_all[0:Nt:int(Nt / num_samples_to_plot)]

    cmap = mpl.colors.ListedColormap(['black', 'goldenrod'])

    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(m_all[0].reshape((height, width)), cmap=cmap)
    plt.show(block=False)

    # draw some data in loop
    for i in range(num_samples_to_plot):
        plt.pause(hold_time)
        im.set_array(m_live[i].reshape((height, width)))
        fig.canvas.draw()


def _set_beta(anneal, annealing_factors, Nt):
    # annealing factors = [self.beta, self.start_beta, self.end_beta, self.growth_factor]
    if anneal == "constant":
        return np.ones(Nt)*annealing_factors[0]
    if anneal == "linear":
        step_size = (annealing_factors[2] - annealing_factors[1]) / Nt
        return np.arange(Nt) * step_size
    if anneal == "geometric":
        all_betas = np.zeros(Nt)
        temp_beta = annealing_factors[1]
        for i in range(Nt):
            all_betas[i] = temp_beta * annealing_factors[3]
        return all_betas


def _cpsl(m, Nm, J, h, Nt, anneal, annealing_factors):
    if anneal == 'constant':
        beta = annealing_factors[0]
        m_all = np.zeros((Nt, Nm))
        for j in range(Nt):
            for i in np.random.permutation(Nm):
                xx = -1 * beta * (np.dot(m, J[:, i]) + h[i])
                m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))
            m_all[j, :] = m
        m_all = np.array(m_all)
        m_all[m_all < 0] = 0
        return m_all, m
    else:
        all_beta = _set_beta(anneal, annealing_factors, Nt)
        m_all = np.zeros((Nt, Nm))
        for j in range(Nt):
            for i in np.random.permutation(Nm):
                xx = -1 * all_beta[j] * (np.dot(m, J[:, i]) + h[i])
                m[i] = np.sign(random.uniform(-1, 1) + np.tanh(xx))
            m_all[j, :] = m
        m_all = np.array(m_all)
        m_all[m_all < 0] = 0
        return m_all, m


@jit(float64[:](float64[:], int64, float64[:], float64[:], float64, int64, float64[:]), nopython=True)
def _cpsl_fast(m, Nm, J, h, beta, Nt, rand_vals):
    m_all = np.zeros(Nt * Nm)  # [[0 for xx in range(a)] for yy in range(b)]
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    h = np.ascontiguousarray(h)
    for j in range(int(Nt)):
        for i in range(int(Nm)):
            xx = -1 * beta * (np.dot(m, J[i * Nm:i * Nm + Nm]) + h[i])
            m[i] = np.sign(rand_vals[j * Nm + i] + np.tanh(xx))
        m_all[j * Nm:j * Nm + Nm] = m
    m_all[m_all < 0] = 0
    # m_all = np.reshape(m_all, (Nt, Nm))
    return m_all


def _ppsl(m, Nm, J, h, beta, Nt, dt, anneal, annealing_factors):
    J = np.array(J)
    h = np.array(h)
    m_all = np.zeros((Nt, Nm))
    h = np.transpose(h)
    for i in range(Nt):
        x = np.multiply(np.add(np.dot(m, J), h), -1 * beta)
        p = np.exp(-1 * dt * np.exp(np.multiply(-1 * m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(Nm))))
        m_all[i, :] = m
    m_all = np.array(m_all)
    m_all[m_all < 0] = 0
    return m_all, m


@jit(float64[:](float64[:], int64, float64[:], float64[:], float64, int64, float64, float64[:]), nopython=True)
def _ppsl_fast(m, Nm, J, h, beta, Nt, dt, randval):
    m_all = np.zeros(Nt * Nm)
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    J = np.reshape(J, (Nm, Nm))
    for i in range(Nt):
        x = np.multiply(np.add(np.dot(J, m), h), -1 * beta)
        p = np.exp(-1 * dt * np.exp(np.multiply(-1 * m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, np.random.rand(Nm))))
        m_all[i * Nm:i * Nm + Nm] = m
    m_all[m_all < 0] = 0
    return m_all


def _dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def _not_drawn(i, j, drawn):
    for ii in drawn:
        if math.isclose(i, ii[0]) and math.isclose(j, ii[1]):
            return False
    return True


