# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:40:42 2018

@author: Christian K
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#get data set

#packages to be used
#glob and os alre used for going through all of the files in a folder
import numpy as np
#TESTING

#PLOTTING 
#import plotly.plotly as py
import glob
import os
#pandas has convenient data structures for uploading and manipulating data
import pandas as pd
from pandas import DataFrame
from pandas import Series
import math

#numpy has a log10() function that I'll use
 
import matplotlib.pyplot as plt
import powerlaw

def F(data): 
#order the data
    data.sort()
    beta = powerlaw.Fit(data)
    datanew = []
    for i in range(len(data)):
        if data[i] > beta:
            datanew.append(data[i])

    N = len(datanew)
    summandone = 0
    for i in range(len(datanew)):
        summandone = summandone + math.log(datanew[i])
    alpha = N/(summandone + -N*math.log(beta))
                    #print(np.log10(max(Pdata)))
                    #plt.show()
                    
    Count = 0
    for i in range(10,1,-1):
        dataoutlier = datanew[(len(datanew)-i):]
        summandtwo = sum(dataoutlier)
        summandthree = sum(datanew)
        T_statistic_empirical = summandtwo/summandthree
        print(len(datanew))
        print(len(dataoutlier))
        print("T stat emp")
        print(T_statistic_empirical)
        
        tstatlist = []
        for j in range(0,50000,1):
            sample = np.random.pareto(alpha,len(datanew)) + beta
            sample.sort()
            summtwo = sum(sample)
            summone = sum(sample[(len(sample)-i):])
            
            T_statistic_null = summone/summtwo
            tstatlist.append(T_statistic_null)
            if j % 10000 == 0:
                print(j/10000)
        
            
        numerator = 0
        print("max t stat")
        print(max(tstatlist))
        print(sum(tstatlist)/len(tstatlist))
        tstatlist.sort()
        print(tstatlist[int(len(tstatlist)/2)])
        for k in range(0,50000,1):
            if tstatlist[k] > T_statistic_empirical:
                numerator = numerator + 1
        P_value = float(numerator/50000)
        print("P val not special is ")
        print(P_value)
        if P_value <= 0.05:
            print("P val is ")
            print(P_value)
            print(i)
            Count = i
            break
        else:
            continue
            
    return P_value, Count, alpha, beta, len(datanew)
        
    


path = '/Users/localadmin/Desktop/Simulation Data/'

Tlist = []
Elist = []
Plist = []
TDKl = []
EDKl = []
PDKl = []

for bet in range(1,2,1):
    bet = 5 
    for filename in glob.glob(os.path.join(path,'*')):
        t = filename.split()
        if len(t) == 3:
            path = filename
            State = False
            for filename2 in glob.glob(os.path.join(path,'*')):
                simulationdata = pd.read_csv(filename+'/Data2.csv')
                if (str(filename2))==(filename+'/Params.csv'):
                    dataparams = pd.read_csv(filename2)
                    betaa = int(10*dataparams.iloc[0][14])
                    if betaa == bet:
                        State = True
                    
                if State == True:
                    Tdata = simulationdata.iloc[:,1]
                    Edata = simulationdata.iloc[:,2]
                    Pdata = simulationdata.iloc[:,3]
                
                    Tdata = Tdata.tolist()
                    Edata = Edata.tolist()
                    Pdata = Pdata.tolist()
#        
#                    fig, ax = plt.subplots()
#                    ax.set_yscale('log',basey=10)
#                    ax.set_xscale('log',basex=10)
#                    ax.set_xlim([37400,1000000])
#                    n, bins, histpatches = ax.hist(Pdata, 50,log='True', facecolor='green', alpha = 0.75)
#                    #print(max(Pdata))
#                    #print(np.log10(max(Pdata)))
#                    #plt.show()
#                    print(bins[2])
                    
                   # Tresult, TDK, alT, betT, lenT = F(Tdata,1)
                    Eresult, EDK, alE, betE, lenE = F(Edata,2)
                    #Presult, PDK, alP, betP, lenP = F(Pdata,3)
        
                    #Tlist.append(Tresult)
                    Elist.append(Eresult)
                    #Plist.append(Presult)
                    #TDKl.append(TDK)
                    EDKl.append(EDK)
                    #PDKl.append(PDK)
        
        
#        fig = plt.figure()
#        ax1 = fig.add_subplot(2,2,1)
#        ax2 = fig.add_subplot(2,2,2)
#        ax3 = fig.add_subplot(2,2,3)
#        
#        ax1 = Tdata.plot(kind='kde',style='k--')
#        ax2 = Edata.plot(kind='kde',style='k--')
#        ax3 = Pdata
        
        

 
#A path for which to find the files in.  Rewrite yours to wherever the files are in
#path = 'C:/Users/kevin/Downloads/Simulation Data'
# 
##os.path.join(path,"*") joins the path you defined with every filename found in the location you specified.
##So here, every "filename" is substituted where "*" is.  So it goes through every Simulation 0, Simulation 1, etc.
##Now we can go through every folder by accessing each folder address
#p =[]
#for filename in glob.glob(os.path.join(path, '*')):
#    #Split the folder name up into a list of items, so that way later on we can used the item with index #=2
#    #as a way to give each data set a unique name
#    t = filename.split()
#    #Here, path is renamed to the folder (Simulation X) we opened inside of the folder
#    path = filename
#    print(filename)
#    #Go through every single file in the Simulation X folder
#    for filename2 in glob.glob(os.path.join(path,'*')):
#        #When we find the file with the data in it...
#        if (str(filename2)==(str(filename)+'\\Alphas.csv')):
#            #Then we store the csv into a DataFrame, a pandas data structure.  All columns from the csv will
#            #also be stored as columns in the DataFrame, data2
#            data = pd.read_csv(filename2)
#            #I picked here the data of interest to me, and put it into a new DataFrame, data
#            #filename3 = str(filename)+'\\Params.csv'
#            #data2 = pd.read_csv(filename3)
#            #if (float(data2.ix[0,14]) == float(0.1))  
#            p.append(float(data.ix[0,2]))
#           
#fig = plt.hist(p,histtype='bar', ec='black')
#plt.title("Histogram for Alpha P, beta = 0.5")
#plt.xlabel("Alpha P")
#plt.ylabel("Frequency")
#           
#plt.savefig('C:/Users/kevin/Downloads/hist.png')
##I clear the figure, just because I'm always worried there will be data that stays on for the next
##iteration.  More of a superstition, I don't think this step is actually necessary
##Clear the list, again, I think this is just a superstition, but a reassuring step anyways
#t = []
##Now that the for loop for the folder of Simulation X is over, my last step is to reset "path" so that the
##next Simulation folder in the original for loop can be acccessed without problems
#path = 'C:/Users/kevin/Downloads/Simulation Data'​