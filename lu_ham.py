import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class LuHamilton(object):
    def __init__(self, shape):
        self.shape = shape
        self.cells = np.zeros(shape, dtype=float)
        self.dA = np.zeros(self.shape)
        self.de = np.zeros(self.shape)

        self.Zc = 3
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
        self.calculate_dA_conv(cells)
        # self.calculate_dA_conv2(cells)
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
        if np.count_nonzero(peaks):
            peaks_sign = np.sign(self.dA * peaks)
            cells_to_redistribute = np.vstack(np.nonzero(peaks)).T
            for cell in cells_to_redistribute:
                self.redistribute(cell, peaks_sign[cell[0], cell[1]])

            # self.de = 0.8 * self.Zc ** 2 * (2 * np.abs(self.dA) / self.Zc - 1) * peaks
            self.de = 0.8 * self.Zc * (np.abs(self.dA)) * peaks
            self.energies.append(np.sum(self.de))
            self.avalanche_time += 1
        else:
            x, y = np.random.randint(self.shape[0]), np.random.randint(self.shape[1])
            self.cells[x, y] += self.drive
            self.cells[0, :], self.cells[-1, :], self.cells[:, 0], self.cells[:, -1] = 0, 0, 0, 0  # keep borders zero
            self.iteration += 1

            if self.avalanche_time:  # these code runs after an avalanche
                self.flare_durations.append(self.avalanche_time)
                self.flare_peak_en.append(np.max(self.energies))
                self.flare_average_en.append(np.sum(self.energies))

                self.avalanche_time = 0
                self.energies = []


if __name__ == '__main__':
    sun = LuHamilton((48, 48))
    iterations = 30000
    import time
    t = time.time()
    for i in range(iterations):
        sun.evolve()

    print("Simulation time: " + str(time.time() - t))

    time_duration_distr = np.bincount(sun.flare_durations)[1:]
    flare_peak_en_disrt = np.bincount((np.array(sun.flare_peak_en)*10).astype(int))
    flare_total_en_disrt = np.bincount((np.array(sun.flare_peak_en)*10).astype(int))

    print(time_duration_distr)
    print(flare_peak_en_disrt)
    print(flare_total_en_disrt)



    fig = plt.figure()
    plt.subplot(311)
    plt.plot(time_duration_distr)
    plt.subplot(312)
    plt.plot(flare_peak_en_disrt)
    plt.subplot(313)
    plt.plot(flare_total_en_disrt)
    plt.show()



