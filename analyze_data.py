import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

filename = 'data/strugarek_nonlocal_con_000.npz'
# filename = 'data/strugarek_con_000.npz'
# filename = 'data/strugarek_con_100.npz'
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





# flare_total_en_disrt = np.bincount(flare_tot_en)

# print(time_duration_distr)
# print(flare_peak_en_disrt)
# print(flare_total_en_disrt)




def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

def calculate_alpha(data, plot=False):
    data_hist = np.bincount(data.astype(int))[1:]
    y = np.log(data_hist)
    y = y[y>0]
    x = np.log(range(1, y.size + 1))

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    if plot:
        plt.plot(x, y, label='data')
        plt.plot(x, f(x, slope, intercept), 'r-', label='Fit line')
        plt.legend(loc='upper right')
        plt.title('Alpha' + str(slope))
        plt.show()

    return slope, intercept


# print calculate_alpha(flare_durations, plot=True)
# print calculate_alpha(flare_peak_en, plot=True)
# print calculate_alpha((flare_peak_en*10).astype(int), plot=True)


def plot_distributions_loglog():
    scale_param = 10
    time_duration_distr = np.bincount(flare_durations.astype(int))
    flare_peak_en_disrt = np.bincount((flare_peak_en * scale_param).astype(int))
    flare_total_en_disrt = np.bincount((flare_tot_en * scale_param).astype(int))


    x_t = range(time_duration_distr.size)
    x_t = np.delete(x_t, np.nonzero(time_duration_distr <= 0)[0])
    time_duration_distr = time_duration_distr[time_duration_distr > 0]
    x_t = np.log(x_t)
    y_t = np.log(time_duration_distr)

    x_pe = range(flare_peak_en_disrt.size)
    x_pe = np.delete(x_pe, np.nonzero(flare_peak_en_disrt <= 0)[0])
    flare_peak_en_disrt = flare_peak_en_disrt[flare_peak_en_disrt > 0]
    x_pe = np.log(x_pe)
    y_pe = np.log(flare_peak_en_disrt)


    x_te = range(flare_total_en_disrt.size)
    x_te = np.delete(x_te, np.nonzero(flare_total_en_disrt <= 0)[0])
    flare_total_en_disrt = flare_total_en_disrt[flare_total_en_disrt > 0]
    x_te = np.log(x_te)
    y_te = np.log(flare_total_en_disrt)


    alpha_t, inter_t, _,_,_ = stats.linregress(x_t, y_t)
    alpha_pe, inter_pe, _,_,_  = stats.linregress(x_pe, y_pe)
    alpha_te, inter_te, _,_,_  = stats.linregress(x_te, y_te)

    print alpha_t, alpha_pe, alpha_te



    fig = plt.figure()
    plt.subplot(131)
    plt.scatter(x_t, y_t, label='Data')
    plt.plot(x_t, f(x_t, alpha_t, inter_t), 'r-', label='Fit line')
    plt.title("Flare durations.alpha=" + str(alpha_t)[:6])
    plt.subplot(132)
    plt.scatter(x_pe, y_pe, label='Data')
    plt.plot(x_pe, f(x_pe, alpha_pe, inter_pe), 'r-', label='Fit line')
    plt.title("Peak Energy.alpha=" + str(alpha_pe)[:6])
    plt.subplot(133)
    plt.scatter(x_te, y_te, label='Data')
    plt.plot(x_te, f(x_te, alpha_te, inter_te), 'r-', label='Fit line')
    plt.title("Total Energy.alpha=" + str(alpha_te)[:6])

    plt.show()


plot_distributions_loglog()
