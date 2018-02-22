import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class Strugarek(object):
    def __init__(self, shape):
        self.shape = shape
        self.cells = np.zeros(shape, dtype=float)
        self.dA = np.zeros(self.shape)
        self.de = np.zeros(self.shape)

        self.Zc = 1    # threshold to lunch avalanche
        self.drive = 1  # additive value to random location
        self.epsilon = 0.00001  # for global driving
        self.iteration_change_drive = 1000     # threshold when to start global driving
        self.sigmaZc = 0.01     # deviation of Zc for stochastic threshold On
        self.Dnc = 0.1          # value for non conservative models

        self.random_redistribution = False
        self.extraction = False
        self.random_threshold = False
        self.conservative = True
        self.local = True

        self.flare_durations = []
        self.flare_peak_en = []
        self.flare_total_en = []

        self.avalanche_time = 0
        self.energies = []

        self.kernel = np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])
        self.iteration = 0
        self.r = self.get_neighbors(self.shape[0])  # assuming that the lattice is square

        self.load_initial_cell_values()


    def load_initial_cell_values(self):
        try:
            init_values = np.load('data/cells_at_soc.npy')
            if init_values.shape == self.shape:
               self.cells = init_values
            else:
               self.iteration_change_drive = 10 ** 8
               print("Initial data is not loaded !!!!!!!!!!!!!")
        except:
            print("Initial data is not loaded !!!!!!!!!!!!!")
            pass

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

        if self.random_threshold:
            Zc_stochastic = np.random.normal(self.Zc, self.sigmaZc, size=self.shape)
            return np.abs(self.dA) > Zc_stochastic
        return np.abs(self.dA) > self.Zc

    def redistribute_LuH(self, cell_xy, peak_sign):
        x, y = cell_xy
        self.cells[x, y] -= peak_sign * self.Zc * 0.8
        self.cells[x + 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x - 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x, y + 1] += peak_sign * self.Zc * 0.2
        self.cells[x, y - 1] += peak_sign * self.Zc * 0.2

    def redistribute_strugarek(self, cell_xy, peak_sign):
            x, y = cell_xy
            Z = peak_sign * self.Zc
            r0 = 1
            if not self.conservative:
                r0 = np.random.uniform(self.Dnc, 1)
            # print np.abs(self.dA[x, y]) - self.Zc
            if self.extraction:
                # print(np.max(np.abs(self.dA[x, y])))
                Z = peak_sign * np.random.uniform(np.abs(self.dA[x, y]) - self.Zc, self.Zc)

            c = r0 * 0.2
            neighbours = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            number_of_neighbors = 4
            if not self.local:
                neighbours = []
                r = np.random.choice(np.arange(1, int(self.shape[0]/2) - 1, dtype=int), size=number_of_neighbors, p=self.r)
                phi = np.random.choice(np.arange(360, dtype=int), size=number_of_neighbors)
                for i in range(number_of_neighbors):
                    x_1, y_1 = int(np.rint(r[i] * np.cos(np.deg2rad(phi[i])))), int(
                        np.rint(r[i] * np.sin(np.deg2rad(phi[i]))))
                    neighbours.append([x_1, y_1])

            self.cells[x, y] -= Z * 0.8
            for x1, y1 in neighbours:
                if self.random_redistribution:
                    r = np.random.rand(4)
                    R = np.sum(r)
                    try:
                        self.cells[x + x1, y + y1] += (r[0] / R) * Z * 4 * c
                    except:
                        pass
                else:
                    try:
                        self.cells[x + x1, y + y1] += Z * c
                    except:
                        pass


    def get_neighbors(self, size):
        r = 1 / (np.arange(1, int(size/2) - 1, dtype=float) ** 2)
        r *= 1 / np.sum(r)  # normalization
        return r



    def evolve(self):
        peaks = self.find_peaks(self.cells)
        if np.count_nonzero(peaks):
            peaks_sign = np.sign(self.dA * peaks)
            cells_to_redistribute = np.vstack(np.nonzero(peaks)).T
            for cell in cells_to_redistribute:
                if self.iteration > self.iteration_change_drive:
                    self.redistribute_strugarek(cell, peaks_sign[cell[0], cell[1]])
                else:
                    self.redistribute_LuH(cell, peaks_sign[cell[0], cell[1]])

            self.de = 0.8 * self.Zc ** 2 * (2 * np.abs(self.dA) / self.Zc - 1) * peaks
            # self.de = 0.8 * self.Zc * (np.abs(self.dA)) * peaks
            E = np.sum(self.de)
            # print(E)
            self.energies.append(E)
            self.avalanche_time += 1
        else:

            # here we change drive from additive to multiplicative
            if self.iteration > self.iteration_change_drive:
                self.cells *= 1 + self.epsilon   # global deterministic driving
            else:
                x, y = np.random.randint(self.shape[0]), np.random.randint(self.shape[1])
                self.cells[x, y] += self.drive
            self.cells[0, :], self.cells[-1, :], self.cells[:, 0], self.cells[:, -1] = 0, 0, 0, 0  # keep borders zero
            self.iteration += 1

            if self.avalanche_time:  # this code runs after an avalanche
                self.flare_durations.append(self.avalanche_time)
                self.flare_peak_en.append(np.max(self.energies))
                self.flare_total_en.append(np.sum(self.energies))

                self.avalanche_time = 0
                self.energies = []

    def save_data(self, filename):
        import os
        if not os.path.exists('data/'):
            os.makedirs('data/')

        np.savez_compressed('data/' + filename,
                            t=np.array(self.flare_durations),
                            pe=np.array(self.flare_peak_en),
                            te=np.array(self.flare_total_en))




if __name__ == '__main__':
    #params
    # size of a lattice
    # threshold to lunch avalanche
    # additive value to random location
    # epsilon fot global driving
    # deviation of Zc for stochastic threshold On
    # value for non conservative models
    # threshold when to start global driving

    # strugarek variations:  random redistribution, random extraction, random threshold, conservation



    params = [[0, 1, 0],   # like in a paper, [random threshold, extraction, random redistr]
              [1, 0, 0],
              [0, 0, 1],
              [1, 1, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1],
              [0, 0, 0],
              ]

    sun = Strugarek((48, 48))



    for p in [params[0]]:
        print(p)
        sun.random_threshold = p[0]
        sun.random_redistribution = p[2]
        sun.extraction = p[1]
        sun.conservative = True
        sun.local = False
        sun.Dnc = 0.1

        iterations = 10**6
        # WARNING check iteration_change_drive and be sure that it is larger then LuHam needs to settle to SOC

        for i in range(iterations):
            sun.evolve()
            if not i % 1000:
                print(float(i)/iterations * 100),
                print('%')

        filename = 'strugarek_local=' + str(sun.local) + '_cons=' + str(sun.conservative) \
                   + '_Dnc=' + str(sun.Dnc) + '_' + str(p[0]) + str(p[1]) + str(p[1]) + '_iter=' + str(iterations)
        sun.save_data(filename)

