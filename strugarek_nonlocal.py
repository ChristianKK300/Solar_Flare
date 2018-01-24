import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class Strugarek(object):
    def __init__(self, shape):
        self.shape = shape
        self.cells = np.zeros(shape, dtype=float)
        # self.set_initial_cell_values()
        self.load_initial_cell_values()
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

        self.flare_durations = []
        self.flare_peak_en = []
        self.flare_total_en = []

        self.avalanche_time = 0
        self.energies = []

        self.kernel = np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])
        self.iteration = 0
        self.r = self.get_neighbors()

    def set_initial_cell_values(self):
        start = 1
        for i in range(24):
            self.cells[i:-i, i:-i] += start

    def load_initial_cell_values(self):
        self.cells = np.load('data/cells_at_soc.npy')

    def calculate_dA_conv(self, cells):
        # convolution = signal.fftconvolve(cells, self.kernel, mode='valid')
        self.dA[1:-1, 1:-1] = signal.fftconvolve(cells, self.kernel, mode='valid')

    def calculate_dA_conv2(self, cells):
        # simple convolution works a bit faster than fft on small dimensions
        self.dA[1:-1, 1:-1] = signal.convolve(cells, self.kernel, mode='valid')


    def find_peaks(self, cells, stochastic_threshold=False):
        # returns cells indices with dA > Zc
        # self.calculate_dA(cells)
        self.calculate_dA_conv(cells)
        # self.calculate_dA_conv2(cells)

        if stochastic_threshold:
            Zc_stochastic = np.random.normal(self.Zc, self.sigmaZc, size=self.shape)
            return np.abs(self.dA) > Zc_stochastic
        return np.abs(self.dA) > self.Zc

    def redistribute_strugarek(self, cell_xy, peak_sign, random_redistribution=False, extraction=False, conservative=True):
            x, y = cell_xy
            Z = peak_sign * self.Zc
            r0 = 1
            if not conservative:
                r0 = np.random.uniform(self.Dnc, 1)
            # print np.abs(self.dA[x, y]) - self.Zc
            if extraction:
                print(np.abs(self.dA[x, y]))
                Z = peak_sign * np.random.uniform(np.abs(self.dA[x, y]) - self.Zc, self.Zc)

            c = r0 * 0.2
            if random_redistribution:
                r = np.random.rand(4)
                R = np.sum(r)
                self.cells[x, y] -= Z * 0.8
                self.cells[x + 1, y] += (r[0] / R) * Z * 4 * c
                self.cells[x - 1, y] += (r[1] / R) * Z * 4 * c
                self.cells[x, y + 1] += (r[2] / R) * Z * 4 * c
                self.cells[x, y - 1] += (r[3] / R) * Z * 4 * c

            else:
                self.cells[x, y] -= Z * 0.8
                self.cells[x + 1, y] += Z * c
                self.cells[x - 1, y] += Z * c
                self.cells[x, y + 1] += Z * c
                self.cells[x, y - 1] += Z * c

    def get_neighbors(self):
        r = 1 / (np.arange(1, 23, dtype=float) ** 2)
        r *= 1/np.sum(r)  # normalization
        return r



    def redistribute_nonlocal(self, cell_xy, peak_sign):
            x, y = cell_xy
            Z = peak_sign * self.Zc
            c = 0.2
            # for each cell to redistribute I need two numbers (x,y)
            number_of_neighbors = 4
            r = np.random.choice(np.arange(1, 23, dtype=int), size=number_of_neighbors, p=self.r)
            phi = np.random.choice(np.arange(360, dtype=int), size=number_of_neighbors)

            self.cells[x, y] -= Z * (1 - c)
            for i in range(number_of_neighbors):
                x1, y1 = int(np.rint(r[i] * np.cos(np.deg2rad(phi[i])))), int(np.rint(r[i] * np.sin(np.deg2rad(phi[i]))))
                # try in case if neighbor out of a lattice
                try:
                    self.cells[x + x1, y + y1] += Z * c
                except:
                    pass

    def evolve(self):
        peaks = self.find_peaks(self.cells, stochastic_threshold=self.random_threshold)
        if np.count_nonzero(peaks):
            peaks_sign = np.sign(self.dA * peaks)
            cells_to_redistribute = np.vstack(np.nonzero(peaks)).T
            for cell in cells_to_redistribute:
                # self.redistribute_strugarek(cell, peaks_sign[cell[0], cell[1]],
                #                             random_redistribution=self.random_redistribution,
                #                             extraction=self.extraction,
                #                             conservative=self.conservative)
                self.redistribute_nonlocal(cell, peaks_sign[cell[0], cell[1]])

            self.de = 0.8 * self.Zc ** 2 * (2 * np.abs(self.dA) / self.Zc - 1) * peaks
            # self.de = 0.8 * self.Zc * (np.abs(self.dA)) * peaks
            E = np.sum(self.de)
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
    # import time
    # from tqdm import tqdm
    #params
    # size of a lattice
    # threshold to lunch avalanche
    # additive value to random location
    # epsilon fot global driving
    # deviation of Zc for stochastic threshold On
    # value for non conservative models
    # threshold when to start global driving

    # strugarek variations:  random redistribution, random extraction, random threshold, conservation

    # problem with random extraction, blows up. Need to tweak drive and threshold params

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
    for p in [params[-1]]:
        print(p)
        sun.random_threshold = p[0]
        sun.random_redistribution = p[2]
        sun.extraction = p[1]
        sun.conservative = True

        iterations = 10**7

        # t = time.time()
        # for i in tqdm(range(iterations)):
        for i in range(iterations):
            sun.evolve()
            if not i % 1000:
                print(float(i)/iterations * 100),
                print('%')
            # print sun.get_neighbors()

        # np.save('data/cells_at_soc', sun.cells)

        # print("Simulation time: " + str(time.time() - t))
        filename = 'strugarek_nonlocal_con1_'+str(p[0])+str(p[1])+str(p[1])+'_iter_' + str(iterations)
        sun.save_data(filename)

