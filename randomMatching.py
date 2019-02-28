#%% Load modules and games
import numpy as np
import copy
import sys
import warnings
import scipy.stats as scst
import scipy as scp
import string
import pickle as pickle
import random
import multiprocessing as mp
import pandas as pd
import time
from sklearn.neighbors import KernelDensity
from IPython import get_ipython
# from general_funs import get_toulouse_games
from math import exp

from pyabc import (ABCSMC, RV, Distribution,
                   PercentileDistanceFunction)
from pyabc.visualization import kde_1d

import pyabc

import os
import tempfile
import matplotlib.pyplot as plt

from RVkde import RVkde, RVmodel, priors_from_posterior


ipython = get_ipython()

import numba
from numba import jit, guvectorize, vectorize, float64, prange         # import the decorator
from numba import int32, float32    # import the types

# Helper functions
@jit(nopython=True)
def gen_matches(p1_order, p2_order):
    np.random.shuffle(p2_order)
    for i in range(len(p2_order)):
        p1_order[p2_order[i]] = i

factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000])

@jit(nopython=True,parallel=False)
def poisson_p(i,τ):
    return (τ**i * np.exp(-τ))/factorials[i]

@jit(nopython=True,parallel=False)
def weighted_rand_int(weigths):
    p = np.random.rand()
    for i in range(len(weigths)):
        if weigths[i] > p:
            return i
        p = p - weigths[i]
    else:
        return len(weigths)

@jit(nopython=True,parallel=False)
def poisson_weight(poisson_vec, strats, j):
    strat = np.zeros_like(strats[0])
    poisson_sum = 0
    for i in range(j):
        strat += poisson_vec[i]*strats[i]
        poisson_sum += poisson_vec[i]
    strat = strat/poisson_sum
    return strat

@jit(nopython=True)
def sample_simplex(strats):
    sample = np.random.rand(len(strats) + 1)
    sample[0] = 0
    sample[-1] = 1.
    sample.sort()
    sample = sample[1:] - sample[0:-1]
    strats[:] = sample[:]


@jit(nopython=True)
def best_reply(self_strats, payoff_mat, opp_s):
    n_strats = payoff_mat.shape[0]
    avg_payoff = payoff_mat @ opp_s
    best_rep = np.zeros(n_strats)
    if (avg_payoff == avg_payoff.max()).sum() == 1:
        best_rep[avg_payoff == avg_payoff.max()] = 1
    else:
        sample = np.zeros((avg_payoff == avg_payoff.max()).sum())
        sample_simplex(sample)
        best_rep[avg_payoff == avg_payoff.max()] = sample
    # best_rep[avg_payoff == avg_payoff.max()] = 1
    # best_rep = best_rep/best_rep.sum()
    self_strats[:] = best_rep[:]

@jit(nopython=True, parallel=False)
def best_reply_logit(payoff_mat, opp_s, λ, strat,  pure=False):
# def best_reply_logit(payoff_mat, opp_s, λ, pure=True):
    n_strats = payoff_mat.shape[0]
    avg_payoff = payoff_mat @ opp_s
    new_strats = np.exp(λ*avg_payoff - (λ*avg_payoff).max())
    new_strats = new_strats/new_strats.sum()
    if pure:
        choice = weighted_rand_int(new_strats)
        new_strats = np.zeros(n_strats)
        new_strats[choice] = 1.
    strat[:] = new_strats[:]


@jit(nopython=True)
def sample_normal_bounded(params_vec, μ_vec, σ_vec, lower_vec, upper_vec):
    for i in range(len(params_vec)):
        params_vec[i] = np.random.normal(μ_vec[i],σ_vec[i])
        while params_vec[i] < lower_vec[i] or params_vec[i] > upper_vec[i]:
            params_vec[i] = np.random.normal(μ_vec[i],σ_vec[i])

@jit(nopython=True)
def draw_beta_from_μσ(μ_in, σ_in, a, b):
    μ = (μ_in-a)/(b-a)
    σ = σ_in/(b-a)
    σ = min(σ, np.sqrt((1-μ)*μ) - 0.000001)
    α = ((1-μ)/σ**2 - 1/μ)*μ**2
    β = α*(1/μ - 1)
    return np.random.beta(α,β)*(b-a) + a

@jit(nopython=True)
def draw_beta(α,β, a, b):
    return np.random.beta(α,β)*(b-a) + a

@jit(nopython=True)
def sample_beta(params_vec, μ_vec, σ_vec, lower_vec, upper_vec):
    for i in range(len(params_vec)):
        params_vec[i] = draw_beta(μ_vec[i],σ_vec[i], lower_vec[i], upper_vec[i])

#######################################################
############ General population functions #############
#######################################################

@jit(nopython=True)
def initiate_params(params, μ_vec, random=True, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([])):
    if not random:
        for i in range(len(params)):
            params[i][:] = μ_vec[:]
    else:
        for i in range(len(params)):
            sample_normal_bounded(params[i], μ_vec, σ_vec, lower_vec, upper_vec)
            # sample_beta(params[i], μ_vec, σ_vec, lower_vec, upper_vec)

@jit
def calc_history(strats, hist):
    for i in range(len(hist)):
        hist[i] = np.mean(strats[:,i])

@jit(nopython=True)
def init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    initiate_params(p1_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random)
    initiate_params(p2_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random)
    if not actual_init:
        init_LPCHM_for(p1_strats,p1_payoffs, p2_payoffs, init_params)
        init_LPCHM_for(p2_strats,p2_payoffs, p1_payoffs, init_params)
    calc_history(p1_strats, p1_history[0])
    calc_history(p2_strats, p2_history[0])

def init_As_and_Ns(self_payoff_mat, self_As, self_Ns):
    self_Ns[:] = 1.
    self_As = np.repeat([self_payoff_mat.mean(axis=1)],len(self_Ns),axis=0)

def create_h_vec(p1_hist, p2_hist, n):
    h_vec = [[p1_hist.copy(), p2_hist.copy()] for i in range(n)]
    return h_vec

@jit(nopython=True)
def flatten_h_jit(hists, flat_hists):
    rounds = len(hists[0][0])
    n_p1 = len(hists[0][0][0])
    n_p2 = len(hists[0][1][0])
    for h in range(len(hists)):
        for r in range(len(hists[0][0])):
            for s in range(len(hists[0][0][0])):
                flat_hists[h][r*n_p1 + s] = hists[h][0][r][s]
            for s in range(len(hists[0][1][0])):
                flat_hists[h][rounds*n_p1 + r*n_p2 + s] = hists[h][1][r][s]


@jit(nopython=True,parallel=False)
def init_LPCHM(self_payoffs, opp_payoffs, params, pure=True, k_rand=True):
    τ = params[0]
    λ = params[1]
    self_n = self_payoffs.shape[0]
    opp_n = opp_payoffs.shape[0]
    if k_rand:
        k = int(np.random.poisson(τ))
        while k > 10:
            k = int(np.random.poisson(τ))
    else:
        k = int(τ)
    self_s = np.ones((k+1, self_n))/(self_n)
    opp_s = np.ones((k+1, opp_n))/(opp_n)
    poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
    if k == 0:
        return self_s[0]
    else:
        for j in range(1, k+1):
            opp_s_guess = poisson_weight(poisson_weights_all, opp_s, j)
            self_s_guess = poisson_weight(poisson_weights_all, self_s, j)
            best_reply_logit(self_payoffs, opp_s_guess, λ, self_s[j], pure=True)
            best_reply_logit(opp_payoffs, self_s_guess, λ, opp_s[j], pure=True)
        return self_s[-1]


@jit(nopython=True, parallel=False)
def init_LPCHM_for(strats, self_payoffs, opp_payoffs, params, pure=True, k_rand=True):
    for i in prange(len(strats)):
        strats[i][:] = init_LPCHM(self_payoffs, opp_payoffs, params, pure=pure, k_rand=k_rand)[:]

#######################################################
###################  LPCHM model   ####################
#######################################################

@jit(nopython=True)
def LPCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, matches, params, opp_guess):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        # p = params[i][0]
        τ = params[i][0]
        λ = params[i][1]
        β = params[i][2]
        match = matches[i]
        opp_guess[i][:] = β*opp_guess[i][:] + opp_s[match][:]
        # if p > np.random.rand():
        k = int(np.random.poisson(τ))
        while k > 10:
            k = int(np.random.poisson(τ))
        if k == 0:
            # self_strats[i][:] = self_s[:]
            self_strats[i][:] = self_strats[i][:]
        else:
            self_s_vec = np.ones((k+1, n_strats))/(n_strats)
            opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
            self_s_vec[0][:] = self_s[:] ## TODO: fix this so it is realistic
            # opp_strats_vec[0][:] = opp_s[match][:]
            opp_strats_vec[0][:] = opp_guess[i][:]/opp_guess[i].sum()
            poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
            for j in range(1, k+1):
                opp_s_guess = poisson_weight(poisson_weights_all, opp_strats_vec, j)
                self_s_guess = poisson_weight(poisson_weights_all, self_s_vec, j)
                if j < k:
                    best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                    best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                else:
                    best_reply_logit(self_payoff_mat, opp_s_guess, λ, self_s_vec[j], pure=True)
                    best_reply_logit(opp_payoff_mat, self_s_guess, λ, opp_strats_vec[j], pure=True)
            self_strats[i][:] = self_s_vec[-1][:]


def run_LPCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, p1_guess, p2_guess, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        LPCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_strats, p2_order, p1_params, p1_guess)
        LPCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_strats_old, p1_order, p2_params, p2_guess)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

def run_LPCHM_osap(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_ind_history, p2_ind_history, p1_matches, p2_matches, p1_payoffs, p2_payoffs, p1_guess, p2_guess, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    p1_ind_history[:,0,:] = p1_strats[:]
    p2_ind_history[:,0,:] = p2_strats[:]
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        LPCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_strats, p2_order, p1_params, p1_guess)
        LPCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_strats_old, p1_order, p2_params, p2_guess)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])
        p1_ind_history[:,i,:] = p1_strats[:]
        p2_ind_history[:,i,:] = p2_strats[:]
        p1_matches[i-1,:] = p1_order[:]
        p2_matches[i-1,:] = p2_order[:]

#######################################################
###################  LBR model   ####################
#######################################################

@jit(nopython=True)
def LBR_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, opp_s, matches, params, opp_guess):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        λ = params[i][1]
        β = params[i][2]
        match = matches[i]
        opp_guess[i][:] = β*opp_guess[i][:] + opp_s[match][:]
        if p > np.random.rand():
            opp_play = opp_guess[i][:]/opp_guess[i].sum()
            best_reply_logit(self_payoff_mat, opp_play, λ, self_strats[i], pure=True)


def run_LBR(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, p1_guess, p2_guess, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        LBR_reply_for(p1_strats, p1_payoffs, p2_payoffs, p2_strats, p2_order, p1_params, p1_guess)
        LBR_reply_for(p2_strats, p2_payoffs, p1_payoffs, p1_strats_old, p1_order, p2_params, p2_guess)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])


def run_LBR_osap(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_ind_history, p2_ind_history, p1_matches, p2_matches, p1_payoffs, p2_payoffs, p1_guess, p2_guess, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    p1_ind_history[:,0,:] = p1_strats[:]
    p2_ind_history[:,0,:] = p2_strats[:]
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        LBR_reply_for(p1_strats, p1_payoffs, p2_payoffs, p2_strats, p2_order, p1_params, p1_guess)
        LBR_reply_for(p2_strats, p2_payoffs, p1_payoffs, p1_strats_old, p1_order, p2_params, p2_guess)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])
        p1_ind_history[:,i,:] = p1_strats[:]
        p2_ind_history[:,i,:] = p2_strats[:]
        p1_matches[i-1,:] = p1_order[:]
        p2_matches[i-1,:] = p2_order[:]



#######################################################
###################  EWA model   ######################
#######################################################

@jit(nopython=True)
def EWA_reply_for(self_strats, self_payoff_mat, self_As, self_Ns, opp_s, matches, params):
    n_strats = self_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        match = matches[i]
        avg_payoffs = self_payoff_mat @ opp_s[match]
        prev_payoff = np.dot(self_strats[i],avg_payoffs)
        p = params[i][0]
        λ = params[i][1]
        φ = params[i][2]
        ρ = params[i][3]
        δ = params[i][4]
        old_N = self_Ns[i]
        self_Ns[i] = 1 + self_Ns[i]*ρ
        for s in range(n_strats):
            self_As[i][s] = (φ*old_N*self_As[i][s] + δ*avg_payoffs[s] + (1-δ)*self_strats[i][s]*prev_payoff)/self_Ns[i]
        if p > np.random.rand():
            new_strats = np.exp(λ*self_As[i] - (λ*self_As[i]).max())
            new_strats = new_strats/new_strats.sum()
            reply = np.zeros(n_strats)
            reply[weighted_rand_int(new_strats)] = 1.
            self_strats[i][:] = reply[:]


def run_EWA(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, p1_As, p2_As, p1_Ns, p2_Ns, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    init_As_and_Ns(p1_payoffs, p1_As, p1_Ns)
    init_As_and_Ns(p2_payoffs, p2_As, p2_Ns)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        EWA_reply_for(p1_strats, p1_payoffs, p1_As, p1_Ns, p2_strats, p2_order, p1_params)
        EWA_reply_for(p2_strats, p2_payoffs, p2_As, p2_Ns, p1_strats_old, p1_order, p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

def run_EWA_osap(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_ind_history, p2_ind_history, p1_matches, p2_matches, p1_payoffs, p2_payoffs, p1_As, p2_As, p1_Ns, p2_Ns, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    init_As_and_Ns(p1_payoffs, p1_As, p1_Ns)
    init_As_and_Ns(p2_payoffs, p2_As, p2_Ns)
    n_ind = len(p2_strats)
    p2_order = np.array(range(n_ind))
    p1_order = np.array(range(n_ind))
    p1_ind_history[:,0,:] = p1_strats[:]
    p2_ind_history[:,0,:] = p2_strats[:]
    for i in range(1, rounds):
        gen_matches(p1_order, p2_order)
        p1_strats_old = copy.deepcopy(p1_strats)
        EWA_reply_for(p1_strats, p1_payoffs, p1_As, p1_Ns, p2_strats, p2_order, p1_params)
        EWA_reply_for(p2_strats, p2_payoffs, p2_As, p2_Ns, p1_strats_old, p1_order, p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])
        p1_ind_history[:,i,:] = p1_strats[:]
        p2_ind_history[:,i,:] = p2_strats[:]
        p1_matches[i-1,:] = p1_order[:]
        p2_matches[i-1,:] = p2_order[:]




#%%
#######################################################
###################  Population object   ######################
#######################################################
class Population:
    """ A population consists of a number of agents who repeatedly plays a game """
    def __init__(self, p1_payoffs, p2_payoffs, rounds, n_p1, n_p2, params_vec=np.array([]), init_params=np.array([1.5,1.]), σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random_params=True, init_strats=False):
        self.p1_payoffs = p1_payoffs
        self.p1_nstrats = p1_payoffs.shape[0]
        self.p2_payoffs = p2_payoffs
        self.p2_nstrats = p2_payoffs.shape[0]
        self.rounds = rounds
        self.params_vec = params_vec
        self.σ_vec = σ_vec
        self.lower_vec = lower_vec
        self.upper_vec = upper_vec
        self.random_params = random_params
        self.p1_params = np.zeros((n_p1, len(params_vec)))
        initiate_params(self.p1_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random_params)
        self.p2_params = np.zeros((n_p2, len(params_vec)))
        initiate_params(self.p2_params, params_vec,  σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random_params)
        self.init_params = init_params
        self.p1_strats = np.zeros((n_p1, self.p1_nstrats))
        self.p2_strats = np.zeros((n_p2, self.p2_nstrats))
        self.p1_guess = np.zeros((n_p2, self.p2_nstrats))
        self.p2_guess = np.zeros((n_p1, self.p1_nstrats))
        self.p1_history = np.zeros((rounds, self.p1_nstrats))
        self.p2_history = np.zeros((rounds, self.p2_nstrats))
        self.p1_ind_history = np.zeros((n_p1, rounds, self.p1_nstrats))
        self.p2_ind_history = np.zeros((n_p2, rounds, self.p2_nstrats))
        self.p1_matches = np.zeros((rounds, n_p1))
        self.p2_matches = np.zeros((rounds, n_p2))
        self.actual_init = False
        if init_strats:
            self.p1_strats = np.repeat([init_strats[0]], n_p1, axis=0)
            self.p2_strats = np.repeat([init_strats[1]], n_p2, axis=0)
            calc_history(self.p1_strats, self.p1_history[0])
            calc_history(self.p2_strats, self.p2_history[0])
            self.actual_init = True
        self.p1_As = np.repeat([p1_payoffs.mean(axis=1)],n_p1,axis=0)
        self.p2_As = np.repeat([p2_payoffs.mean(axis=1)],n_p1,axis=0)
        self.p1_Ns = np.ones(n_p1)
        self.p2_Ns = np.ones(n_p2)

    def run_EWA(self):
        run_EWA(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.p1_As, self.p2_As, self.p1_Ns, self.p2_Ns, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]

    def mul_runs_EWA(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_EWA(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.p1_As, self.p2_As, self.p1_Ns, self.p2_Ns, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_EWA_osap(self):
        run_EWA_osap(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_ind_history, self.p2_ind_history, self.p1_matches, self.p2_matches, self.p1_payoffs, self.p2_payoffs, self.p1_As, self.p2_As, self.p1_Ns, self.p2_Ns, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return {"pop_hists": [self.p1_history, self.p2_history], "ind_hists": [self.p1_ind_history, self.p2_ind_history], "matches": [self.p1_matches, self.p2_matches], "params":[self.p1_params, self.p2_params]}


    def run_LPCHM(self):
        run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]

    def mul_runs_LPCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_LPCHM_osap(self):
        run_LPCHM_osap(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_ind_history, self.p2_ind_history, self.p1_matches, self.p2_matches, self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return {"pop_hists": [self.p1_history, self.p2_history], "ind_hists": [self.p1_ind_history, self.p2_ind_history], "matches": [self.p1_matches, self.p2_matches], "params":[self.p1_params, self.p2_params]}

    def run_LBR(self):
        run_LBR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]

    def mul_runs_LBR(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LBR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_LBR_osap(self):
        run_LBR_osap(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_ind_history, self.p2_ind_history, self.p1_matches, self.p2_matches, self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return {"pop_hists": [self.p1_history, self.p2_history], "ind_hists": [self.p1_ind_history, self.p2_ind_history], "matches": [self.p1_matches, self.p2_matches], "params":[self.p1_params, self.p2_params]}


#######################################################
################### Function for analysis  ############
#######################################################

def sample_fun(dist):
    np.random.seed()
    if isinstance(dist, tuple):
        return np.random.uniform(low=dist[0], high=dist[1])
    else:
        return np.random.choice(dist)

def flatten_h(hists):
    return np.array([np.append(np.array(hist[0]).ravel(), np.array(hist[1]).ravel()) for hist in hists])

def flatten_single_hist(hist):
    return np.append(np.array(hist[0]).ravel(), np.array(hist[1]).ravel())

def gen_kde(simulated, bw=0.07):
    flat = flatten_h(simulated)
    kde = KernelDensity(bandwidth=bw)
    kde.fit(flat)
    return kde

def gen_kde_from_flat(simulated, bw=0.07):
    kde = KernelDensity(bandwidth=bw)
    kde.fit(simulated)
    return kde

def calc_from_kde(kde, hist):
    # flat_actual = np.array([flatten_single_hist(hist)])
    res = kde.score(hist)
    if np.isnan(res):
        res = np.finfo('d').min
    return res

def calc_avg_from_kde(kde, hists):
    tot_ll = 0
    for hist in hists:
        tot_ll += calc_from_kde(kde, hist)
    return(tot_ll/len(hists))

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def calc_likelihood(simulated_flat, hists_flat):
    simulated = unflatten_data(simulated_flat, shape)
    hists = unflatten_data(hists_flat, shape)
    tot_score = 0
    for sim, hist in zip(simulated, hists):
        kde = KernelDensity(bandwidth=bw)
        kde.fit(sim)
        tot_score += kde.score(hist)
    return tot_score

def get_gid_data(gids):
    return {pseudo:{gid:pseudo_data_correct[pseudo][gid] for gid in gids} for pseudo in pseudo_data_correct}

def flatten_data(hists):
    shape_vec = []
    hists_vec = np.array([])
    for hist in hists:
        shape_vec.append(hist.shape)
        hists_vec = np.append(hists_vec, hist.flatten())
    return(hists_vec,shape_vec)

def flat_data_from_osap(res):
    hists = []
    for gid in np.sort(list(res.keys())):
        hists.append(flatten_h(res[gid]["pop_hists"]))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def unflatten_data(hists_vec, shape_vec):
    start = 0
    end = 0
    hists = []
    for shape in shape_vec:
        end += shape[0]*shape[1]
        hists.append(np.reshape(hists_vec[start:end],shape))
        start += shape[0]*shape[1]
    return hists

def plot_models(history):
    for i in history.alive_models(history.max_t):
        df, w = history.get_distribution(m=i)
        df_copy = df.copy()
        model = model_names[i]
        for param in αβ_params[model]:
            a, b = αβ_lims[model][param]
            df[param] = df_copy.apply(lambda x: st.beta.mean(x[param], x[param+"_sd"])*(b-a) + a, axis=1)
            df[param+"_sd"] = df_copy.apply(lambda x: st.beta.std(x[param], x[param+"_sd"])*(b-a), axis=1)
        plot = plot_kde_matrix(df, w, limits=αβ_param_spaces[model_names[i]])
        plot.savefig("fig/SSE/abc-beta" + str(n_particles) + "-" + str(max_pops)+ "-bw" + str(bw) + "-"  + pseduo_pop  + "-" + model_names[i] + ".png")
    plt.close("all")

rgb_cols = [(228/255, 26/255, 28/255), (55/255, 126/255, 184/255), (77/255, 175/255, 74/255), (152/255, 78/255, 163/255), (255/255, 127/255, 0/255)]

def plot_h(h, save=False):
    plt.figure(figsize=(24,8))
    for role in range(2):
        ax = plt.subplot(1,2,(role + 1))
        ax.set_ylim([-0.01,1.01])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        n_s = h[role].shape[1]
        plt.title("Player role " + str(role + 1))
        for s in range(n_s):
            plt.plot(h[role][:,s], color=rgb_cols[s], ls="-", label="Strat "+str(s))
        ax.legend()
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def run_model_return_flat_hist(model, params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params=False):
    data = model(params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params=random_params)
    # data = model(params, random_params=False)
    hists = unflatten_data(data["data"], data["shape"])
    return hists

# def likelihood_from_sim(model, test_res, params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params=False, cores=mp.cpu_count(), bw=0.05):
#     pool = mp.Pool(processes=cores)
#     res_mp = [pool.apply_async(run_model_return_flat_hist, args=(model, params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params)) for x in range(cores)]
#     res_unflattened = [p.get() for p in res_mp]
#     pool.close()
#     pool.join()
#     kde_hists = [np.vstack([x[i] for x in res_unflattened]) for i in range(len(gids))]
#     test_hists = unflatten_data(test_res["flat_hists"], test_res["shape"])
#     ll = 0
#     for i in range(len(gids)):
#         kde_01 = gen_kde_from_flat(kde_hists[i], bw=0.5)
#         ll += calc_from_kde(kde_01, test_hists[i])
#     return ll
#     return kde_hists

def likelihood_from_sim(model, test_res_vec, params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params=False, bws=[0.07, 0.05, 0.03, 0.01], cores=mp.cpu_count()):
    pool = mp.Pool(processes=cores)
    res_mp = [pool.apply_async(run_model_return_flat_hist, args=(model, params, gids, games, default_init, rounds, n_per_core, p1_size, p2_size, random_params)) for x in range(cores)]
    res_unflattened = [p.get() for p in res_mp]
    pool.close()
    pool.join()
    kde_hists = [np.vstack([x[i] for x in res_unflattened]) for i in range(len(gids))]
    test_lls_vec = [[0]*len(bws) for i in range(len(test_res_vec))]
    for k in range(len(bws)):
        for gid in range(len(gids)):
            bw = bws[k]
            kde_01 = gen_kde_from_flat(kde_hists[gid], bw=bw)
            for i in range(len(test_res_vec)):
                test_hists = unflatten_data(test_res_vec[i]["flat_hists"], test_res_vec[i]["shape"])
                test_lls_vec[i][k] += calc_from_kde(kde_01, test_hists[gid])
        # ll_vec.append(ll)
    return test_lls_vec
    # test_hists = unflatten_data(test_res["flat_hists"], test_res["shape"])
    # ll_vec = []
    # for bw in bws:
    #     ll = 0
    #     for i in range(len(gids)):
    #         kde_01 = gen_kde_from_flat(kde_hists[i], bw=bw)
    #         ll += calc_from_kde(kde_01, test_hists[i])
    #     ll_vec.append(ll)
    # test_lls_vec.append(ll_vec)
    # return ll_vec


#######################################################
################### ABC functions  ####################
#######################################################

def distance(x,y):
    simulated = unflatten_data(x["data"], x["shape"])
    hists = unflatten_data(y["data"], y["shape"])
    tot_distance = 0
    for sim, hist in zip(simulated, hists):
        tot_distance += scp.spatial.distance.euclidean(sim,hist)
    return tot_distance



def LPCHM_model(parameters, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True):
    # if (len(parameters) == 3) and (not ("β" in parameters)):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        τ = parameters[0]
        λ = parameters[1]
        β = parameters[2]
        β_sd = 0.01
        τ_sd = 0.01
        λ_sd = 0.01
        init_params=default_init
    else:
        τ = parameters["τ"]
        λ = parameters["λ"]
        β = parameters["β"]
        if "β_sd" in parameters.keys():
            τ_sd = parameters["τ_sd"]
            λ_sd = parameters["λ_sd"]
            β_sd = parameters["β_sd"]
        else:
            τ_sd = 0.01
            λ_sd = 0.01
            β_sd = 0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    hists = []
    for gid in gids:
        pop_LPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[τ, λ, β], σ_vec=[τ_sd, λ_sd, β_sd], lower_vec=[0., 0., 0.], upper_vec=[4., 10., 1.], random_params=random_params)
        hists.append(flatten_h(pop_LPCHM.mul_runs_LPCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def LPCHM_osap(parameters, gids, games, default_init, rounds, p1_size, p2_size, random_params=True):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        τ = parameters[0]
        λ = parameters[1]
        β = parameters[2]
        β_sd = 0.01
        τ_sd = 0.01
        λ_sd = 0.01
        init_params=default_init
    else:
        β = parameters["β"]
        τ = parameters["τ"]
        λ = parameters["λ"]
        if "β_sd" in parameters.keys():
            β_sd = parameters["β_sd"]
            τ_sd = parameters["τ_sd"]
            λ_sd = parameters["λ_sd"]
        else:
            β_sd = 0.01
            τ_sd = 0.01
            λ_sd = 0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    res_dict = dict()
    hists = []
    for gid in gids:
        pop_LPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[τ, λ, β], σ_vec=[τ_sd, λ_sd, β_sd], lower_vec=[0., 0., 0.], upper_vec=[4., 10., 1.], random_params=random_params)
        res_dict[gid] = pop_LPCHM.run_LPCHM_osap()
        hists.append(flatten_h([res_dict[gid]["pop_hists"]]))
    flat_hists, shape = flatten_data(hists)
    res_dict["flat_hists"] = flat_hists
    res_dict["shape"] = shape
    return res_dict



def LBR_model(parameters, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        p = parameters[0]
        λ = parameters[1]
        β = parameters[2]
        β_sd = 0.01
        p_sd = 0.01
        λ_sd = 0.01
        init_params=default_init
    else:
        p = parameters["p"]
        λ = parameters["λ"]
        β = parameters["β"]
        if "β_sd" in parameters.keys():
            p_sd = parameters["p_sd"]
            λ_sd = parameters["λ_sd"]
            β_sd = parameters["β_sd"]
        else:
            p_sd = 0.01
            λ_sd = 0.01
            β_sd = 0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    hists = []
    for gid in gids:
        pop_LBR = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p, λ, β], σ_vec=[p_sd, λ_sd, β_sd], lower_vec=[0., 0., 0.], upper_vec=[4., 10., 1.], random_params=random_params)
        hists.append(flatten_h(pop_LBR.mul_runs_LBR(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def LBR_osap(parameters, gids, games, default_init, rounds, p1_size, p2_size, random_params=True):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        p = parameters[1]
        λ = parameters[2]
        β = parameters[0]
        p_sd = 0.01
        λ_sd = 0.01
        β_sd = 0.01
        init_params=default_init
    else:
        p = parameters["p"]
        λ = parameters["λ"]
        β = parameters["β"]
        if "β_sd" in parameters.keys():
            p_sd = parameters["p_sd"]
            λ_sd = parameters["λ_sd"]
            β_sd = parameters["β_sd"]
        else:
            p_sd = 0.01
            λ_sd = 0.01
            β_sd = 0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    res_dict = dict()
    hists = []
    for gid in gids:
        pop_LBR = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p, λ, β], σ_vec=[p_sd, λ_sd, β_sd], lower_vec=[0., 0., 0.], upper_vec=[4., 10., 1.], random_params=random_params)
        res_dict[gid] = pop_LBR.run_LBR_osap()
        hists.append(flatten_h([res_dict[gid]["pop_hists"]]))
    flat_hists, shape = flatten_data(hists)
    res_dict["flat_hists"] = flat_hists
    res_dict["shape"] = shape
    return res_dict

def EWA_model(parameters, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True):
    # if (len(parameters) == 5) and (not ("p" in parameters)):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        p = parameters[0]
        λ = parameters[1]
        φ = parameters[2]
        ρ = parameters[3]
        δ = parameters[4]
        p_sd, λ_sd, φ_sd, ρ_sd, δ_sd = 0.01, 0.01, 0.01, 0.01,0.01
        init_params=default_init
    else:
        p = parameters["p"]
        λ = parameters["λ"]
        φ = parameters["φ"]
        ρ = parameters["ρ"]
        δ = parameters["δ"]
        if "p_sd" in parameters.keys():
            p_sd = parameters["p_sd"]
            λ_sd = parameters["λ_sd"]
            φ_sd = parameters["φ_sd"]
            ρ_sd = parameters["ρ_sd"]
            δ_sd = parameters["δ_sd"]
        else:
            p_sd, λ_sd, φ_sd, ρ_sd, δ_sd = 0.01, 0.01, 0.01, 0.01,0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    hists = []
    for gid in gids:
        pop_EWA = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, init_params=init_params, params_vec=[p, λ, φ, ρ, δ], σ_vec=[p_sd, λ_sd, φ_sd, ρ_sd, δ_sd], lower_vec=[0.,0., 0., 0., 0.], upper_vec=[1.,10., 1., 1., 1.], random_params=random_params)
        hists.append(flatten_h(pop_EWA.mul_runs_EWA(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def EWA_osap(parameters, gids, games, default_init, rounds, p1_size, p2_size, random_params=True):
    # if (len(parameters) == 5) and (not ("p" in parameters)):
    if isinstance(parameters, list) or isinstance(parameters, np.ndarray):
        p = parameters[0]
        λ = parameters[1]
        φ = parameters[2]
        ρ = parameters[3]
        δ = parameters[4]
        p_sd, λ_sd, φ_sd, ρ_sd, δ_sd = 0.01, 0.01, 0.01, 0.01,0.01
        init_params=default_init
    else:
        p = parameters["p"]
        λ = parameters["λ"]
        φ = parameters["φ"]
        ρ = parameters["ρ"]
        δ = parameters["δ"]
        if "p_sd" in parameters.keys():
            p_sd = parameters["p_sd"]
            λ_sd = parameters["λ_sd"]
            φ_sd = parameters["φ_sd"]
            ρ_sd = parameters["ρ_sd"]
            δ_sd = parameters["δ_sd"]
        else:
            p_sd, λ_sd, φ_sd, ρ_sd, δ_sd = 0.01, 0.01, 0.01, 0.01,0.01
        if "init_τ" in parameters:
            init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
        else:
             init_params=default_init
    res_dict = dict()
    hists = []
    for gid in gids:
        pop_EWA = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, init_params=init_params, params_vec=[p, λ, φ, ρ, δ], σ_vec=[p_sd, λ_sd, φ_sd, ρ_sd, δ_sd], lower_vec=[0.,0., 0., 0., 0.], upper_vec=[1.,10., 1., 1., 1.], random_params=random_params)
        res_dict[gid] = pop_EWA.run_EWA_osap()
        hists.append(flatten_h([res_dict[gid]["pop_hists"]]))
    flat_hists, shape = flatten_data(hists)
    res_dict["flat_hists"] = flat_hists
    res_dict["shape"] = shape
    return res_dict


def abc_from_data(y, model_names, models, priors, n_particles, init_ε, α, max_pops, min_accept_rate, add_meta_info, model_prior=None):
    if model_prior == None:
        model_prior = RV("randint", 0, len(model_names))
    abc = ABCSMC(models, priors, distance, model_prior=model_prior, population_size=n_particles, eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=init_ε, alpha=α))
    db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp-pseudos.db"))
    meta_info = {"distribution":"Trunc Normal"}
    meta_info.update(add_meta_info)
    abc.new(db_path, y, meta_info=meta_info)
    history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_pops, min_accept_rate=min_accept_rate)
    return(history)

def abc_from_res_separate(res, gids, model_names, gid_models_org, priors, param_spaces, n_particles, init_ε, α, max_pops, min_accept_rate, add_meta_info):
    dfs = dict()
    ws = dict()
    model_prior = RV("randint", 0, len(model_names))
    priors = copy.deepcopy(priors)
    gid_models = copy.deepcopy(gid_models_org)
    model_names = copy.deepcopy(model_names)
    param_spaces = copy.deepcopy(param_spaces)
    for gid in gids:
        data = np.array([flatten_single_hist(res[gid]["pop_hists"])])
        flatten_single_hist(res[gid]["pop_hists"])
        shape = [data.shape]
        y = {"data": data, "shape":shape}
        meta_info = {"distribution":"Trunc Normal", "gid":gid}
        meta_info.update(add_meta_info)
        abc_hist = abc_from_data(y, model_names, gid_models[gid], priors, n_particles, init_ε, α, max_pops, min_accept_rate, meta_info, model_prior=model_prior)
        mod_probs = abc_hist.get_model_probabilities()
        print(mod_probs)
        print("actual_params: ",np.mean(res[1]["params"][0], axis=0))
        ml_m = mod_probs.get_values()[abc_hist.max_t].argmax()
        best_model = model_names[ml_m]
        print("True:", add_meta_info["model"], " Est:", best_model)
        est_df, est_w = abc_hist.get_distribution(ml_m)
        print("Est params: ", ml_from_abc(est_df, est_w))
        # print("Est params: ", median_from_abc(est_df, est_w))
        dfs[gid] = []
        ws[gid] = []
        alive_models = abc_hist.alive_models(abc_hist.max_t)
        for m in alive_models:
            df, w = abc_hist.get_distribution(m)
            dfs[gid].append(df)
            ws[gid].append(w)

        model_names = [model_names[m] for m in alive_models]
        for g in gids:
            gid_models[g] = [gid_models[g][m] for m in alive_models]

        priors = [priors_from_posterior(dfs[gid][m], ws[gid][m], param_spaces[model_names[m]]) for m in range(len(alive_models))]
        model_prior = RVmodel(abc_hist)
        best_model
    return (dfs, ws, best_model)


def abc_from_res(res, gids, model_names, models_wrap, priors, param_spaces, n_particles, init_ε, α, max_pops, min_accept_rate, add_meta_info):
    dfs = dict()
    ws = dict()
    model_prior = RV("randint", 0, len(model_names))
    priors = copy.deepcopy(priors)
    # gid_models = copy.deepcopy(gid_models_org)
    model_names = copy.deepcopy(model_names)
    param_spaces = copy.deepcopy(param_spaces)
    # for gid in gids:
    #     data = np.array([flatten_single_hist(res[gid]["pop_hists"])])
    #     flatten_single_hist(res[gid]["pop_hists"])
    #     shape = [data.shape]
    y = {"data": res["flat_hists"], "shape":res["shape"]}
    meta_info = {"distribution":"Trunc Normal"}
    meta_info.update(add_meta_info)
    abc_hist = abc_from_data(y, model_names, models_wrap, priors, n_particles, init_ε, α, max_pops, min_accept_rate, meta_info, model_prior=model_prior)
    mod_probs = abc_hist.get_model_probabilities()
    end_mod_probs = mod_probs.get_values()[abc_hist.max_t]
    print(mod_probs)
    print("actual_params: ",np.mean(res[1]["params"][0], axis=0))
    ml_m = mod_probs.get_values()[abc_hist.max_t].argmax()
    best_model = model_names[ml_m]
    print("True:", add_meta_info["model"], " Est:", best_model)
    est_df, est_w = abc_hist.get_distribution(ml_m)
    print("Est params: ", ml_from_abc(est_df, est_w))
    # print("Est params: ", median_from_abc(est_df, est_w))
    alive_models = abc_hist.alive_models(abc_hist.max_t)

    for m in alive_models:
        df, w = abc_hist.get_distribution(m)
        dfs[m] = df
        ws[m] = w

    return (dfs, ws, best_model, end_mod_probs)


def ml_from_abc(df, w):
    params = {}
    for key in df.columns:
        x, pdf = kde_1d(df, w, key)
        i = np.argmax(pdf)
        params[key] = x[i]
    return params

def median_from_abc(df, w):
    params = {}
    for key in df.columns:
        x, pdf = kde_1d(df, w, key, numx=100)
        cdf = 0
        i = 0
        while cdf < 0.5:
            cdf += (x[i+1] - x[i])*pdf[i]
            i += 1
        params[key] = x[i]
    return params

def mean_from_abc(df, w):
    params = {}
    for key in df.columns:
        x, pdf = kde_1d(df, w, key, numx=100)
        mean = 0
        for i in range(len(x)-1):
            mean += (x[i+1] - x[i])*pdf[i]*(x[i+1] + x[i])/2
        params[key] = mean
    return params
