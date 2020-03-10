import random
import numpy as np
import turtle
import math
from numba import jit, float64, int64
import cupy as cp
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

    def getModel(self):
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

        # establish an annealing factors array which contains all annealing related information to be sent to cpsl/ppsl
        annealing_factors = np.asarray([self.beta, self.start_beta, self.end_beta, self.growth_factor])

        # jit requires specific input types
        jit_beta = float64(self.beta)
        jit_m = float64(self.m)
        jit_J = np.ndarray.flatten(np.ndarray.astype(self.J, "float64"))
        jit_h = np.ndarray.flatten(np.ndarray.astype(self.h, "float64"))
        jit_Nm = np.int64(self.Nm)
        jit_rand = np.random.uniform(-1, 1, Nt * self.Nm)


        if model is None:
            model = self.model
        if model == "cpsl":
            if not gpu:
                m_all = _cpsl_fast(jit_m, jit_Nm, jit_J, jit_h, Nt, jit_rand, self.beta)
                m_all = np.reshape(m_all, (Nt, self.Nm))
                self.m = m_all[Nt - 1, :]
            else:
                # gpu requires sending variables
                NmNtbeta = np.array([self.Nm, Nt, self.beta], dtype=np.int)
                cpu_samples = np.zeros((Nt, self.Nm))
                rand_vals = jit_rand

                gpu_NmNtbeta = cp.asarray(NmNtbeta)
                gpu_samples = cp.asarray(cpu_samples)
                gpu_J = cp.asarray(self.J)
                gpu_h = cp.asarray(self.h)
                gpu_m = cp.asarray(self.m)
                gpu_rand_vals = cp.asarray(rand_vals)

                _cpsl_core_gpu(cpu_samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals)
                m_all = cpu_samples
                self.m = m_all[Nt - 1, :]
        elif model == "ppsl":
            if not gpu:
                m_all = _ppsl_fast(jit_m, jit_Nm, jit_J, jit_h, jit_beta, Nt, self.dt, jit_rand)
                m_all = np.reshape(m_all, (Nt, self.Nm))
                self.m = m_all[Nt - 1, :]
            else:
                # gpu requires sending variables

                NmNtbeta = np.array([self.Nm, Nt, self.beta], dtype=np.int)
                samples = np.zeros((Nt, self.Nm))
                rand_vals = jit_rand

                gpu_NmNtbeta = cp.asarray(NmNtbeta)
                gpu_samples = cp.asarray(samples)
                gpu_J = cp.asarray(self.J)
                gpu_h = cp.asarray(self.h)
                gpu_m = cp.asarray(self.m)
                gpu_rand_vals = cp.asarray(rand_vals)
                _ppsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals)
                m_all = samples
                self.m = m_all[Nt - 1, :]
        else:
            print("Error: unknown model")

        if ret_base == 'samples' or ret_base == 'sample' or ret_base == 'b' or ret_base == 'binary' or ret_base == 2:
            return m_all
        elif ret_base == 'decimal' or ret_base == 'd' or ret_base == 'deci' or ret_base == 'decimals' or ret_base == 10:
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
        orig = Image.open(file_name)
        baw = orig.convert('1') #convert to black and white (baw)
        width, height = baw.size
        if width != height:
            print("ERROR: image must be a square")
            return
        J_temp = np.zeros((height * width, height * width))

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
                    J_temp[cur, right] = 1
                elif col + 1 < width and baw.getpixel((col, row)) == baw.getpixel((col + 1, row)):
                    J_temp[cur, right] = -1
                if row + 1 < height and baw.getpixel((col, row)) != baw.getpixel((col, row + 1)):
                    J_temp[cur, down] = 1
                elif row + 1 < width and baw.getpixel((col, row)) == baw.getpixel((col, row + 1)):
                    J_temp[cur, down] = -1
                if col - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col - 1, row)):
                    J_temp[cur, left] = 1
                elif col - 1 < width and baw.getpixel((col, row)) == baw.getpixel((col - 1, row)):
                    J_temp[cur, left] = -1
                if row - 1 > 0 and baw.getpixel((col, row)) != baw.getpixel((col, row - 1)):
                    J_temp[cur, up] = 1
                elif row - 1 < width and baw.getpixel((col, row)) == baw.getpixel((col, row - 1)):
                    J_temp[cur, up] = -1
        h = np.zeros(width*height)

        self.J = np.array(J_temp)
        self.h = np.array(h)
        self.Nm = np.int64(self.J.shape[0])
        self.reset() # sets up an initial m state

    def load(self, name=None, city_graph=None, tsp_modifier=None):
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
        elif name == 'tsp':
            """
            build a j matrix for some travelling salesman problem graph. 

            designing a travelling salesman J:
            Rule 1: 1 between pbits of same city
            Rule 2: 1 between pbits of same order
            Rule 3: negative distances as weights between rows (ex. all p-bits in city-1-row to all cities in city-3-row)
            Rule 4: 0 connections from city_n-order_n to itself
            
            See excel sheet (building_J_for_tsp in shark tank 2020 purdue onedrive)
            """

            city_graph = np.asarray(city_graph)
            # city_graph = city_graph / city_graph.max()

            # normalize the graph between 0 and 1
            # city_graph = np.add((city_graph - np.min(city_graph)) / np.ptp(city_graph), 0.01)
            city_graph = np.divide(city_graph, np.amax(np.abs(city_graph)))
            # city_graph = np.divide(1,city_graph) # make the largest weight smallest and smallest weight largest so smallest weight pulls together most

            if tsp_modifier is None:
                tsp_modifier = 1
            Nm_cities = len(city_graph[0])
            self.J = np.zeros((Nm_cities ** 2, Nm_cities ** 2))

            # Rule 3: negative distances from one city to another
            for i in range(Nm_cities):
                for j in range(Nm_cities):
                    #if order(i) is one away from order(j)
                    if i == j:
                        continue
                    off_diag = np.ones(Nm_cities-1) * city_graph[j, i]
                    #set both off diagonals (one to the left and right of main diagonal) of weights to off_diag
                    weights_i_j = np.diag(off_diag, 1) + np.diag(off_diag, -1)

                    self.J[j * Nm_cities: j * Nm_cities + Nm_cities, i * Nm_cities: i * Nm_cities + Nm_cities] = weights_i_j[:,:]

            # Rule 1 - 1 between pbits of same city
            for i in range(Nm_cities):
                self.J[i * Nm_cities:i * Nm_cities + Nm_cities, i * Nm_cities:i * Nm_cities + Nm_cities] = tsp_modifier # dif

            # Rule 2 - 1 between pbits of same order
            for i in range(Nm_cities ** 2):
                for j in range(Nm_cities ** 2):
                    if i == j % Nm_cities or j == i % Nm_cities:
                        self.J[i, j] = tsp_modifier

            # Rule 4: 0s on the diagonal
            np.fill_diagonal(self.J, 0)

            self.Nm = len(self.J)
            self.h = np.zeros(self.Nm)
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
        return m_all
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
        return m_all


@jit(float64[:](float64[:], int64, float64[:], float64[:], int64, float64[:], float64), nopython=True)
def _cpsl_fast(m, Nm, J, h, Nt, rand_vals, beta):
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
    return m_all

def _cpsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals):
    for j in range(int(gpu_NmNtbeta[1])):
        for i in range(int(gpu_NmNtbeta[0])):
            gpu_temp = -1 * gpu_NmNtbeta[2] * (cp.add(cp.dot(gpu_m, gpu_J[i, :]), gpu_h[i]))
            gpu_m[i] = cp.sign(gpu_rand_vals[j*gpu_NmNtbeta[0]+i] + cp.tanh(gpu_temp))
        gpu_samples[j,:] = gpu_m
    samples[:,:] = cp.asnumpy(gpu_samples)

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
    return m_all


@jit(float64[:](float64[:], int64, float64[:], float64[:], float64, int64, float64, float64[:]), nopython=True)
def _ppsl_fast(m, Nm, J, h, beta, Nt, dt, randval):
    m_all = np.zeros(Nt * Nm)
    m = np.ascontiguousarray(m)
    J = np.ascontiguousarray(J)
    J = np.reshape(J, (Nm, Nm))
    for i in range(Nt):
        x = np.multiply(np.add(np.dot(J, m), h), -1 * beta)
        p = np.exp(-1 * dt * np.exp(np.multiply(-1 * m, x)))
        m = np.multiply(m, np.sign(np.subtract(p, randval[i * Nm:i*Nm+Nm])))
        m_all[i * Nm:i * Nm + Nm] = m
    m_all[m_all < 0] = 0
    return m_all

def _ppsl_core_gpu(samples, gpu_samples, gpu_m, gpu_NmNtbeta, gpu_J, gpu_h, gpu_rand_vals):
    for i in range(int(gpu_NmNtbeta[1])):
        x = -1 * cp.add(cp.dot(gpu_m, gpu_J), gpu_h)
        p = cp.exp(-1 * 1 / 6 * cp.exp(cp.multiply(-1* gpu_m, x)))
        gpu_m = cp.multiply(gpu_m, cp.sign(cp.add(p, -1 * gpu_rand_vals[int(i * gpu_NmNtbeta[0]): int(i * gpu_NmNtbeta[0] + gpu_NmNtbeta[0])])))
        gpu_samples[i, :] = gpu_m
    samples[:, :] = cp.asnumpy(gpu_samples)


def _dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def _not_drawn(i, j, drawn):
    for ii in drawn:
        if math.isclose(i, ii[0]) and math.isclose(j, ii[1]):
            return False
    return True


