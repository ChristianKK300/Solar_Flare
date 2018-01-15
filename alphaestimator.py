import numpy as np

def get_alpha(data):
        data = np.array(data)
        xmin = np.min(data)
        summand1 = 0
        for dt in data:
            summand1 = summand1 + np.log(dt/xmin)
        #logsum = sum(np.log(data))
        alpha = 1 + (data.size) * (summand1) ** (-1)
        sigma = (alpha - 1)/((data.size)**(0.5)) + (1/data.size)
        return alpha, sigma
        print(alpha)
        print(sigma)