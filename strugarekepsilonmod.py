import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

class Strugarek(object):
    def __init__(self, shape):
        self.shape = shape
        self.cells = np.zeros(shape, dtype=float)
        self.dA = np.zeros(self.shape)
        self.de = np.zeros(self.shape)

        self.Zc = 2    # threshold to lunch avalanche
        self.drive = 0.05  # additive value to random location
        self.epsilon = 0.000001  # for global driving
        self.iteration_change_drive = 10000     # threshold when to start global driving
        self.sigmaZc = 0.05 # deviation of Zc for stochastic threshold On
        self.Dnc = 0.8      # value for non conservative models
        #0.7,0.6,0.5,...

        self.random_redistribution = False
        self.extraction = False
        self.random_threshold = False
        self.conservative = True

        self.flare_durations = []
        self.flare_peak_en = []
        self.flare_average_en = []

        self.avalanche_time = 0
        self.energies = []

        self.kernel = np.array([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])
        self.iteration = 0



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



    def find_peaks(self, cells, stochastic_threshold=False):
        # returns cells indices with dA > Zc
        # self.calculate_dA(cells)
        # self.calculate_dA_conv(cells)
        self.calculate_dA_conv2(cells)

        if stochastic_threshold:
            Zc_stochastic = np.random.normal(self.Zc, self.sigmaZc, size=self.shape)
            return np.abs(self.dA) > Zc_stochastic
        return np.abs(self.dA) > self.Zc

    def redistribute(self, cell_xy, peak_sign):
        x, y = cell_xy
        self.cells[x, y] -= peak_sign * self.Zc * 0.8
        self.cells[x + 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x - 1, y] += peak_sign * self.Zc * 0.2
        self.cells[x, y + 1] += peak_sign * self.Zc * 0.2
        self.cells[x, y - 1] += peak_sign * self.Zc * 0.2

    def redistribute_strugarek(self, cell_xy, peak_sign, random_redistribution=False, extraction=False, conservative=True):
            x, y = cell_xy
            Z = peak_sign * self.Zc
            r0 = 1
            if not conservative:
                r0 = np.random.uniform(self.Dnc, 1)
            # print np.abs(self.dA[x, y]) - self.Zc
            if extraction:
                #print(np.abs(self.dA[x, y]))
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



    def evolve(self):
        peaks = self.find_peaks(self.cells, stochastic_threshold=self.random_threshold)
        if np.count_nonzero(peaks):
            peaks_sign = np.sign(self.dA * peaks)
            cells_to_redistribute = np.vstack(np.nonzero(peaks)).T
            for cell in cells_to_redistribute:
                self.redistribute_strugarek(cell, peaks_sign[cell[0], cell[1]],
                                            random_redistribution=self.random_redistribution,
                                            extraction=self.extraction,
                                            conservative=self.conservative)

            # self.de = 0.8 * self.Zc ** 2 * (2 * np.abs(self.dA) / self.Zc - 1) * peaks
            self.de = 0.8 * self.Zc * (np.abs(self.dA)) * peaks
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

            if self.avalanche_time:  # these code runs after an avalanche
                self.flare_durations.append(self.avalanche_time)
                self.flare_peak_en.append(np.max(self.energies))
                self.flare_average_en.append(np.sum(self.energies))

                self.avalanche_time = 0
                self.energies = []

    def save_data(self):
        filename = 'strugarek_redistr_' + str(self.random_redistribution)
        # todo: implement saving data with numpy savez




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

    # problem with random extraction, blows up. Need to tweak drive and threshold params

#    sun = Strugarek((48, 48))
#    sun.random_redistribution = False
#    sun.extraction = False
#    sun.random_threshold = False
#    sun.conservative = True
#
#    iterations = 10000000
#    import time
#    t = time.time()
#    for i in range(iterations):
#        sun.evolve()
#
#
#    print("Simulation time: " + str(time.time() - t))
#
#
#    sun.save_data()
#
#    time_duration_distr = np.bincount(sun.flare_durations)[1:]
#    flare_peak_en_disrt = np.bincount((np.array(sun.flare_peak_en)*10).astype(int))
#    flare_total_en_disrt = np.bincount((np.array(sun.flare_peak_en)*10).astype(int))
#
#    # print(time_duration_distr)
#    # print(flare_peak_en_disrt)
#    # print(flare_total_en_disrt)
#
#    fig = plt.figure()
#    plt.subplot(311)
#    plt.plot(time_duration_distr)
#    plt.subplot(312)
#    plt.plot(flare_peak_en_disrt)
#    plt.subplot(313)
#    plt.plot(flare_total_en_disrt)
#    plt.show()
#    
#    sun2 = Strugarek((48, 48))
#    sun2.random_redistribution = True
#    sun2.extraction = False
#    sun2.random_threshold = False
#    sun2.conservative = True
#
#    iterations = 10000000
#    import time
#    t = time.time()
#    for i in range(iterations):
#        sun2.evolve()
#
#
#    print("Simulation time: " + str(time.time() - t))
#
#
#    sun2.save_data()
#
#    time_duration_distr2 = np.bincount(sun2.flare_durations)[1:]
#    flare_peak_en_disrt2 = np.bincount((np.array(sun2.flare_peak_en)*10).astype(int))
#    flare_total_en_disrt2 = np.bincount((np.array(sun2.flare_peak_en)*10).astype(int))
#
#    # print(time_duration_distr)
#    # print(flare_peak_en_disrt)
#    # print(flare_total_en_disrt)
#
##    fig = plt.figure()
##    plt.subplot(311)
##    plt.plot(time_duration_distr2)
##    plt.subplot(312)
##    plt.plot(flare_peak_en_disrt2)
##    plt.subplot(313)
##    plt.plot(flare_total_en_disrt2)
##    plt.show()
#
#    sun3 = Strugarek((48, 48))
#    sun3.random_redistribution = False
#    sun3.extraction = True
#    sun3.random_threshold = False
#    sun3.conservative = True
#
#    iterations = 10000000
#    import time
#    t = time.time()
#    for i in range(iterations):
#        sun3.evolve()
#
#
#    print("Simulation time: " + str(time.time() - t))
#
#
#    sun3.save_data()
#
#    time_duration_distr3 = np.bincount(sun3.flare_durations)[1:]
#    flare_peak_en_disrt3 = np.bincount((np.array(sun3.flare_peak_en)*10).astype(int))
#    flare_total_en_disrt3 = np.bincount((np.array(sun3.flare_peak_en)*10).astype(int))
#
##---------
#
#    sun4 = Strugarek((48, 48))
#    sun4.random_redistribution = False
#    sun4.extraction = False
#    sun4.random_threshold = True
#    sun4.conservative = True
#
#    iterations = 10000000
#    import time
#    t = time.time()
#    for i in range(iterations):
#        sun4.evolve()
#
#
#    print("Simulation time: " + str(time.time() - t))

#
#    sun4.save_data()
#
#    time_duration_distr4 = np.bincount(sun4.flare_durations)[1:]
#    flare_peak_en_disrt4 = np.bincount((np.array(sun4.flare_peak_en)*10).astype(int))
#    flare_total_en_disrt4 = np.bincount((np.array(sun4.flare_peak_en)*10).astype(int))
    
    #---------

    sun5 = Strugarek((48, 48))
    sun5.random_redistribution = False
    sun5.extraction = False
    sun5.random_threshold = False
    sun5.conservative = False

    iterations = 10000000
    import time
    t = time.time()
    for i in range(iterations):
        sun5.evolve()


    print("Simulation time: " + str(time.time() - t))


    sun5.save_data()

    time_duration_distr5 = np.bincount(sun5.flare_durations)[1:]
    flare_peak_en_disrt5 = np.bincount((np.array(sun5.flare_peak_en)*10).astype(int))
    flare_total_en_disrt5 = np.bincount((np.array(sun5.flare_average_en)*10).astype(int))
    
        #---------

    sun6 = Strugarek((48, 48))
    sun6.random_redistribution = True
    sun6.extraction = False
    sun6.random_threshold = False
    sun6.conservative = False

    iterations = 10000000
    import time
    t = time.time()
    for i in range(iterations):
        sun6.evolve()


    print("Simulation time: " + str(time.time() - t))


    sun6.save_data()

    time_duration_distr6 = np.bincount(sun6.flare_durations)[1:]
    flare_peak_en_disrt6 = np.bincount((np.array(sun6.flare_peak_en)*10).astype(int))
    flare_total_en_disrt6 = np.bincount((np.array(sun6.flare_average_en)*10).astype(int))
    
            #---------

    sun7 = Strugarek((48, 48))
    sun7.random_redistribution = False
    sun7.extraction = True
    sun7.random_threshold = False
    sun7.conservative = False

    iterations = 10000000
    import time
    t = time.time()
    for i in range(iterations):
        sun7.evolve()


    print("Simulation time: " + str(time.time() - t))


    sun7.save_data()

    time_duration_distr7 = np.bincount(sun7.flare_durations)[1:]
    flare_peak_en_disrt7 = np.bincount((np.array(sun7.flare_peak_en)*10).astype(int))
    flare_total_en_disrt7 = np.bincount((np.array(sun7.flare_average_en)*10).astype(int))
    
                #---------

    sun8 = Strugarek((48, 48))
    sun8.random_redistribution = False
    sun8.extraction = False
    sun8.random_threshold = True
    sun8.conservative = False

    iterations = 10000000
    import time
    t = time.time()
    for i in range(iterations):
        sun8.evolve()


    print("Simulation time: " + str(time.time() - t))


    sun8.save_data()

    time_duration_distr8 = np.bincount(sun8.flare_durations)[1:]
    flare_peak_en_disrt8 = np.bincount((np.array(sun8.flare_peak_en)*10).astype(int))
    flare_total_en_disrt8 = np.bincount((np.array(sun8.flare_average_en)*10).astype(int))
    
                    #---------
