#%% Load modules and games
import numpy as np
import copy
import sys
import warnings
import scipy.stats as scst
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



ipython = get_ipython()

import numba
from numba import jit, guvectorize, vectorize, float64, prange         # import the decorator
from numba import int32, float32    # import the types
# %load_ext line_profiler



# games = get_toulouse_games()
# for gid in games:
#     games[gid][0] = games[gid][0].astype(float)
#     games[gid][1] = games[gid][1].astype(float)

# np.set_printoptions(precision=3, suppress=True)

### jit funcitons

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

@jit(nopython=True)
def LPCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, matches, params, self_guess):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        # p = params[i][0]
        τ = params[i][0]
        λ = params[i][1]
        β = params[i][2]
        match = matches[i]
        self_guess[i][:] = β*self_guess[i][:] + opp_s[match][:]
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
            self_s_vec[0][:] = self_s[:]
            # opp_strats_vec[0][:] = opp_s[match][:]
            opp_strats_vec[0][:] = self_guess[i][:]/self_guess[i].sum()
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


# @jit(nopython=True)
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

# @jit(nopython=True)

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



#%%
### Population stuff
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

    def run_LPCHM(self):
        run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]

    def mul_runs_LPCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.p1_guess, self.p2_guess, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    # def mul_runs_model(self, model, n):
    #     if model == "BR":
    #         return self.mul_runs_BR(n)
    #     elif model == "LK":
    #         return self.mul_runs_LK(n)
    #     elif model == "LPCHM":
    #         return self.mul_runs_LPCHM(n)
    #     elif model == "EWA":
    #         return self.mul_runs_EWA(n)
    #     else:
    #         print("Invalid model too mul_runs_model")

#%%

# mturk_games = dict()
# mturk_games[1] = [np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]]),np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]])]
# mturk_games[2] = [np.array([[9.,0.],[0.,3.],[6.,6.]]), np.array([[3.,0.,0.],[0.,9.,0.]])]
# mturk_games[3] = [np.array([[2.,2.,4.,4.],[8.,8.,2.,2.],[0.,2.,0.,2.],[6.,0.,6.,0.]]), np.array([[2.,4.,2.,4.],[2.,4.,8.,2.],[8.,2.,2.,4.],[8.,2.,8.,2.]])]
# mturk_games[4] = [np.array([[2.,2.,2.,2.,2.],[1.,4.,4.,4.,4.],[1.,3.,10.,10.,10.],[1.,3.,5.,18.,18.],[1.,3.,5.,7.,30.]]), np.array([[0.,3.,3.,3.,3.],[0.,1.,7.,7.,7.],[0.,1.,4.,13.,13.],[0.,1.,4.,6.,23.],[0.,1.,4.,6.,8.]])]
# mturk_games[5] = [np.array([[12.,4.,0.],[4.,12.,0.],[0.,14.,2.],[6.,6.,6.]]), np.array([[12.,4.,0.,0.], [4.,12.,0.,0.],[14.,0.,2.,0.]])]

# games = mturk_games
# gid = 5
# rounds = 30
# p1_size = 20
# p2_size = 20
# init_params = [1.5, 1.0]

# parameters = {'λ': 1.7, 'λ_sd': 0.2, 'p': 0.7, 'p_sd': 0.1, 'φ': 0.9, 'φ_sd': 0.1, 'ρ': 0.21, 'ρ_sd': 0.1, 'δ': 0.5, 'δ_sd': 0.1}
# p = parameters["p"]
# p_sd = parameters["p_sd"]
# λ = parameters["λ"]
# λ_sd = parameters["λ_sd"]
# φ = parameters["φ"]
# φ_sd = parameters["φ_sd"]
# ρ = parameters["ρ"]
# ρ_sd = parameters["ρ_sd"]
# δ = parameters["δ"]
# δ_sd = parameters["δ_sd"]
# pop_EWA = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, init_params=init_params, params_vec=[p, λ, φ, ρ, δ], σ_vec=[p_sd, λ_sd, φ_sd, ρ_sd, δ_sd], lower_vec=[0.,0., 0., 0., 0.], upper_vec=[1.,10., 1., 1., 1.], random_params=True)
# pop_EWA.run_EWA()

# parameters = {"τ": 0.7, "τ_sd":0.1, "λ":4.23, "λ_sd":0.3, "β":0.9, "β_sd":0.1}

# β = parameters["β"]
# τ = parameters["τ"]
# λ = parameters["λ"]
# β_sd = parameters["β_sd"]
# τ_sd = parameters["τ_sd"]
# λ_sd = parameters["λ_sd"]
# pop_LPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[τ, λ, β], σ_vec=[τ_sd, λ_sd, β_sd], lower_vec=[0., 0., 0.], upper_vec=[4., 10., 1.], random_params=True)
# pop_LPCHM.run_LPCHM()

#%% Analyze funcs
# def flatten_h(hists):
#     flat_hist = np.zeros((len(hists), len(hists[0][0])*(len(hists[0][0][0]) + len(hists[0][1][0]))))
#     flatten_h_jit(hists,flat_hist)
#     return flat_hist


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
    flat_actual = np.array([flatten_single_hist(hist)])
    res = kde.score(flat_actual)
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
