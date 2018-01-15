import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class LuHamilton(object):
    def __init__(self, shape):
        self.shape = shape
        self.cells = np.zeros(shape, dtype=float)
        self.dA = np.zeros(self.shape)
        self.de = np.zeros(self.shape)

        self.Zc = 2
        self.drive = 1

        self.flare_durations = []
        self.flare_peak_en = []
        self.flare_average_en = []

        self.avalanche_time = 0
        self.energies = []

        self.kernel = np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])
        self.iteration = 0
        self.iteration_change_drive = 100000
        self.epsilon = 0.001

    def calculate_dA(self, cells):
        self.dA *= 0
        for (i, j), cell in np.ndenumerate(cells):
            if 0 < i < self.shape[0] - 1 and 0 < j < self.shape[1] - 1:   # boundary conditions
                neighbors = cells[i+1, j] + cells[i-1, j] + cells[i, j+1] + cells[i, j-1]
                self.dA[i, j] = cell - neighbors/4

    def calculate_dA_conv(self, cells):
        # convolution = signal.fftconvolve(cells, self.kernel, mode='valid')
        self.dA[1:-1, 1:-1] = signal.fftconvolve(cells, self.kernel, mode='valid')

    def calculate_dA_conv2(self, cells):
        # simple convolution works a bit faster than fft on small dimensions
        self.dA[1:-1, 1:-1] = signal.convolve(cells, self.kernel, mode='valid')



    def find_peaks(self, cells):
        # returns cells indices with dA > Zc
        # self.calculate_dA(cells)
        # self.calculate_dA_conv(cells)
        self.calculate_dA_conv2(cells)
        return np.abs(self.dA) > self.Zc

    def redistribute(self, cell_xy, peak_sign):
        x, y = cell_xy
        self.cells[x, y] -= peak_sign * self.Zc * 0.8
        self.cells[x + 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x - 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x, y + 1] += peak_sign * self.Zc * 0.2
        self.cells[x, y - 1] += peak_sign * self.Zc * 0.2

    def evolve(self):
        peaks = self.find_peaks(self.cells)
        peaks_sign = np.sign(self.dA * peaks)
        if np.count_nonzero(peaks):
            cells_to_redistribute = np.vstack(np.nonzero(peaks)).T
            for cell in cells_to_redistribute:
                self.redistribute(cell, peaks_sign[cell[0], cell[1]])

            # self.de = 0.8 * self.Zc ** 2 * (2 * np.abs(self.dA) / self.Zc - 1) * peaks
            self.de = 0.8 * self.Zc * (np.abs(self.dA)) * peaks
            E = np.sum(self.de)
            self.energies.append(E)
            self.avalanche_time += 1
        else:
            x, y = np.random.randint(self.shape[0]), np.random.randint(self.shape[1])
            # here we change drive from additive to multiplicative
            if self.iteration > self.iteration_change_drive:
                self.cells[x,y] *= 1 + self.epsilon
            else:
                self.cells[x, y] += self.drive
            self.cells[0, :], self.cells[-1, :], self.cells[:, 0], self.cells[:, -1] = 0, 0, 0, 0  # keep borders zero
            self.iteration += 1

            if self.avalanche_time:  # these code runs after an avalanche
                self.flare_durations.append(self.avalanche_time)
                self.flare_peak_en.append(np.max(self.energies))
                self.flare_average_en.append(np.mean(self.energies))

                self.avalanche_time = 0
                self.energies = []

    def get_alpha(self, data):
        data = np.array(data)
        xmin = np.min(data)
        summand1 = 0
        for dt in data:
            summand1 = summand1 + np.log(dt/xmin)
        alpha = 1 + (data.size) * (summand1) ** (-1)
        sigma = (alpha - 1)/((data.size)**(0.5)) + (1/data.size)
        print alpha
        print sigma


if __name__ == '__main__':
    sun = LuHamilton((48, 48))
    iterations = 300000
    import time
    t = time.time()
    for i in range(iterations):
        sun.evolve()
        # print sun.dA

    print "Simulation time: " + str(time.time() - t)

    time_duration_distr = np.bincount(sun.flare_durations)[1:]
    print time_duration_distr
    print np.bincount((np.array(sun.flare_peak_en)*10).astype(int))
    print np.bincount((np.array(sun.flare_average_en)*10).astype(int))

    from scipy.optimize import curve_fit

    def func_powerlaw(x, m, c, c0):
        return c0 + x**(-m) * c

    X = range(1, time_duration_distr.size + 1)
    # popt, pcov = curve_fit(func_powerlaw, X, time_duration_distr, p0 = np.asarray([1,1,0]))
    # print popt, pcov

    # import powerlaw
    # results = powerlaw.Fit(time_duration_distr)
    # print(results.power_law.alpha)
    # print(results.power_law.xmin)
    # R, p = results.distribution_compare('power_law', 'lognormal')

    # quit()
    # plt.figure(figsize=(10, 5))
    # plt.plot(X, func_powerlaw(X, results.power_law.alpha, 350, 0), '--')
    # plt.plot(X, time_duration_distr, 'ro')
    # plt.legend()
    # plt.show()
    # quit()

    fig = plt.figure()
    plt.subplot(311)
    plt.plot(np.bincount(sun.flare_durations)[1:])
    plt.subplot(312)
    plt.plot(np.bincount((np.array(sun.flare_peak_en)*10).astype(int))[1:])
    plt.subplot(313)
    plt.plot(np.bincount((np.array(sun.flare_average_en)*10).astype(int))[1:])
    plt.show()


    # print sun.flare_durations
    # print np.array(sun.flare_peak_en)*10
    # print sun.flare_average_en


