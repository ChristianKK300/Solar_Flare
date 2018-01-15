import numpy as np
import powerlaw

def get_alpha(data):
        data = np.array(data)
        
        result = powerlaw.Fit(data)
        xminimum = result.power_law.xmin
        #xmin from power law package
        xminimum = xminimum - 0.5
        summand1 = 0
        datanew = []
        #alpha2 is the estimation from powerlaw package
        alpha2 = result.power_law.alpha
        #xmin the smallest value in data
        xminimum = min(data)
        for i in range(len(data)):
            if data[i] > xminimum:
                datanew.append(data[i])
        data = datanew
        for dt in data:
            summand1 = summand1 + np.log(dt/xminimum)
        #logsum = sum(np.log(data))
        #alpha is the estimation from this function
        alpha = 1 + len(data) * (summand1) ** (-1)
        sigma = (alpha - 1)/(len(data)**(0.5)) + (1/(len(data)))
        print(alpha)
        print(sigma)
        print(xminimum)
        return alpha, sigma, alpha2
