# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:09:04 2017

@author: escriva
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols


#Working inside subdirectory
abspath = os.path.abspath(__file__)
absname = os.path.dirname(abspath)
os.chdir(absname)

data = pd.read_csv('overdraft.csv')
surface = pd.read_csv('surfacesupplies.csv')

data = data[data.year<2004]

c2mean = []
c2var = []
cvmean = []
cvvar = []
combmean = []
combvar = []
c2stdmean = []
cvstdmean = []
combstdmean = []


for i in np.arange(1,22):
    name1 = "c2vsim"+str(i)
    name2 = "cvhm"+str(i)
    combined = data[name1]
    combined = combined.append(data[name2])
    c2stats  = (-data[name1].mean(), data[name1].var())
    cvstats  = (-data[name2].mean(), data[name2].var())
    combstats = (-combined.mean(), combined.var())
    c2mean.append(c2stats[0])
    c2var.append(c2stats[1])
    cvmean.append(cvstats[0])
    cvvar.append(cvstats[1])
    combmean.append(combstats[0])
    combvar.append(combstats[1])
    c2stdmean.append(np.sqrt(c2stats[1])/np.sqrt(len(data)))
    cvstdmean.append(np.sqrt(cvstats[1])/np.sqrt(len(data)))
    combstdmean.append(np.sqrt(combstats[1])/np.sqrt(len(data)))
    
del combined, name1, name2, c2stats, cvstats, combstats

def probendoverdraft(initialoverdraft, years, mean, variance, reduse):
    cumvariance = years * variance
    standdev = np.sqrt(cumvariance)
    return norm.cdf(0,initialoverdraft-(years*(reduse-mean)),standdev)

def overdraftgivenpercentile(initialoverdraft, years, mean, variance, reduse, percentile):
    cumvariance = years * variance
    standdev = np.sqrt(cumvariance)
    return norm.ppf(percentile,initialoverdraft-(years*(reduse-mean)),standdev)
  
perc = [0.01, 0.1,0.25,0.5,0.75,0.9, 0.99]
b = np.zeros((11,7))
#plotting normal dist function (i should be the subregion)
fig, ax = plt.subplots()
for i in np.arange(9,21):
    mu = combmean[i]
    muc2 = c2mean[i]
    mucv = cvmean[i]
    sigma = combstdmean[i]
    sigmac2 = c2stdmean[i]
    sigmacv = cvstdmean[i]
    minx = min(mu-3.5*sigma, muc2-3.5*sigmac2,mucv-3.5*sigmacv)
    maxx = max(mu+3.5*sigma, muc2+3.5*sigmac2,mucv+3.5*sigmacv)
    x= np.linspace(minx, maxx,100)
    regionname = "region" + str(i+1)
    """
    plt.plot(x,mlab.normpdf(x,mu,sigma))
    plt.plot(x,mlab.normpdf(x,muc2,sigmac2))
    plt.plot(x,mlab.normpdf(x,mucv,sigmacv))
    """
    
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        initialoverdraft = 0
        years = 20
        variance = combvar[i]
        a.append(probendoverdraft(initialoverdraft,years,mu,variance,j))
    
    
    """
    plt.plot(xax, a,label=regionname)
    plt.text(xax[len(xax)/2],a[len(a)/2],'{i}'.format(i=i+1))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,600])
    plt.xlabel('Reduction in water use (taf/year)',fontsize=14)
    plt.ylabel('Probability of achieving sustainability after 200 years',fontsize=14)
    plt.yticks(np.arange(0,1.05,0.1))
    """
    """    
        for k in np.arange(0,7):
            if np.around(probendoverdraft(100,20,mu,variance,j),decimals=2)==perc[k]:
                b[i-10][k]=j
    
    
    plt.plot(b[i-10],perc,label = regionname)
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,600])
    plt.scatter(0.5,b[i-10][3],marker='o')
    plt.legend(loc='4')
    plt.yticks(np.arange(0,1.05,0.1))
    """

#Plotting boot strap    
random2 = []
change = float(1000)/float(1945000)/float(0.2)
for i in np.arange(20000):
    variance = combvar[20]
    randomseries=[]
    randomseries2 = []
    randomseries.append(0)
    for j in np.arange(1,21):
        randomseries.append(randomseries[j-1]+np.random.normal(0,np.sqrt(variance)))
    for j in np.arange(21,26):
        randomseries.append(randomseries[j-1]+np.random.normal(0*0.75,np.sqrt(variance)))
    for j in np.arange(26,31):
        randomseries.append(randomseries[j-1]+np.random.normal(0*0.5,np.sqrt(variance)))
    for j in np.arange(31,36):
        randomseries.append(randomseries[j-1]+np.random.normal(0*0.25,np.sqrt(variance)))
    for j in np.arange(36,41):
        randomseries.append(randomseries[j-1]+np.random.normal(0,np.sqrt(variance)))
    for j in np.arange(0,41):
        randomseries2.append(randomseries[j])
    randomseries2 = randomseries2
    random2.append(randomseries2)
sns.tsplot(random2, err_style='boot_traces',n_boot=500)