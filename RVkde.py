#%%
import numpy as np
import scipy.stats as scst
import scipy as scp
import pandas as pd


from pyabc import (ABCSMC, RV, RVBase, Distribution,
                   PercentileDistanceFunction)
from pyabc.transition import MultivariateNormalTransition
import pyabc
import copy


class RVkde(RVBase):
    def __init__(self, df, w, key, min, max):
        self.df = df
        self.w = w
        self.key = key
        self.kde =  MultivariateNormalTransition(scaling=1)
        self.kde.fit(df[[key]], w)
        self.min = min
        self.max = max

        # Calucating truncated away cdf to compensate in pdf
        min_xs = np.linspace(self.min - 10., self.min, num=200)
        min_pdfs = [self.kde.pdf(pd.DataFrame({self.key:[x]})) for x in min_xs]
        min_cdf = np.sum(min_pdfs)*10/200

        max_xs = np.linspace(self.max, self.max + 10, num=200)
        max_pdfs = [self.kde.pdf(pd.DataFrame({self.key:[x]})) for x in max_xs]
        max_cdf = np.sum(max_pdfs)*10/200
        self.trunc_cdf = min_cdf + max_cdf

    def rvs(self):
        x = self.kde.rvs()
        while x[0] < self.min or x[0] > self.max:
            x = self.kde.rvs()
        return(x[0])

    def pdf(self,x):
        p = 0.
        if x > self.min and x < self.max:
            x = pd.DataFrame({self.key:[x]})
            p = self.kde.pdf(x)
        return p/(1-self.trunc_cdf)

    def copy(self):
        return copy.deepcopy(self)

    def pmf(self):
        return 0.

    def cdf(self, x):
        cdf = 0
        if x > self.min:
            xs = np.linspace(self.min, x, num=100)
            pdfs = [self.pdf(x) for x in xs]
            cdf = np.sum(pdfs)*(x - self.min)/100
        return cdf

class RVmodel(RVBase):
    def __init__(self, abc_hist):
        self.n_models = len(abc_hist.alive_models(abc_hist.max_t))
        self.mod_probs = abc_hist.get_model_probabilities()
        self.p_vec = self.mod_probs.get_values()[abc_hist.max_t]
        try:
            self.p_vec = [self.p_vec[m] for m in range(len(self.p_vec))]
        except:
            print(self.p_vec)
            print([m for m in range(len(self.p_vec))])
        self.p_vec = self.p_vec/np.sum(self.p_vec)

    def rvs(self):
        p = np.random.rand()
        for i in range(self.n_models):
            if self.p_vec[i] > p:
                return i
            p = p - self.p_vec[i]
        else:
            return (self.n_models -1)

    def pdf(self):
        return 0.

    def copy(self):
        return copy.deepcopy(self)

    def pmf(self, m):
        return self.p_vec[m]

    def cdf(self, x):
        cdf = 0
        for i in range(m):
            cdf += p_vec[i]
        return cdf


def priors_from_posterior(df,w, p_space):
    priors = dict()
    for key in df.columns:
        priors[key] = RVkde(df, w, key, p_space[key][0], p_space[key][1])
    return Distribution(**priors)
