import numpy as np
import matplotlib.pyplot as plt


filename = 'data/strugarek_con_000.npz'
data = np.load(filename)

# cells = np.load('data/cells_at_soc.npy')
# print np.max(cells)
# print cells[:5,:5]
# plt.imshow(cells)
# plt.show()


# print np.load('data/cells_at_soc.npy')

flare_durations = data['t'][20:]
flare_peak_en = data['pe'][20:]
flare_tot_en = data['te'][20:]



time_duration_distr = np.bincount(flare_durations.astype(int))[1:]
flare_peak_en_disrt = np.bincount((flare_peak_en*10).astype(int))
flare_total_en_disrt = np.bincount((flare_tot_en*10).astype(int))
# flare_total_en_disrt = np.bincount(flare_tot_en)

# print(time_duration_distr)
# print(flare_peak_en_disrt)
# print(flare_total_en_disrt)
def plot_distributions_xy():
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(time_duration_distr)
    plt.subplot(312)
    plt.plot(flare_peak_en_disrt)
    plt.subplot(313)
    plt.plot(flare_total_en_disrt)
    plt.show()

def plot_distributions_loglog():
    t_range = np.log(range(1, time_duration_distr.size + 1))
    pe_range = np.log(range(1, flare_peak_en_disrt.size + 1))
    te_range = np.log(range(1, flare_total_en_disrt.size + 1))
    fig = plt.figure()
    plt.subplot(321)
    plt.plot(t_range, np.log(time_duration_distr))
    plt.subplot(323)
    plt.plot(pe_range, np.log(flare_peak_en_disrt))
    plt.subplot(325)
    plt.plot(te_range, np.log(flare_total_en_disrt))

    plt.subplot(322)
    plt.plot(time_duration_distr)
    plt.subplot(324)
    plt.plot(flare_peak_en_disrt)
    plt.subplot(326)
    plt.plot(flare_total_en_disrt)
    plt.show()

plot_distributions_loglog()