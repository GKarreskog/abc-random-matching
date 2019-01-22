#%% Load functions
import randomMatching
from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap, EWA_reply_for, weighted_rand_int
from importlib import reload
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

# ABC stuff
from pyabc.visualization import plot_kde_matrix
from pyabc.visualization import plot_kde_1d, kde_1d
from pyabc.transition import MultivariateNormalTransition
import pyabc.random_variables as rv


import os
import tempfile

import scipy.stats as st
import scipy as scp


from pyabc import (ABCSMC, RV, Distribution,
                   PercentileDistanceFunction)

import pyabc

import numba
from numba import jit, guvectorize, vectorize, float64, prange         # import the decorator
from numba import int32, float32    # import the types

#%%
reload(randomMatching)
from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap, EWA_reply_for, weighted_rand_int

#%% Some setup
@jit(nopython=True)
def osap_EWA_pred(self_strats, self_payoff_mat, self_As, self_N, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    avg_payoffs = self_payoff_mat @ opp_s
    prev_payoff = np.dot(self_strats,avg_payoffs)
    p = params[0]
    λ = params[1]
    φ = params[2]
    ρ = params[3]
    δ = params[4]
    old_N = self_N
    self_N = 1 + self_N * ρ
    for s in range(n_strats):
        self_As[s] = (φ*old_N*self_As[s] + δ*avg_payoffs[s] + (1-δ)*self_strats[s]*prev_payoff)/self_N
        if p > np.random.rand():
            new_strats = np.exp(λ*self_As - (λ*self_As).max())
            new_strats = new_strats/new_strats.sum()
            self_strats[:] = new_strats[:]



#%%


sample_param_spaces = dict()
sample_param_spaces["EWA"] = {"λ":(1.,8.), "λ_sd":(0,1.), "p":(0.3, 0.9), "p_sd":(0,0.2), "φ":(0., 1.), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}
sample_param_spaces["LPCHM"] = {"τ":(0.2, 1.8), "τ_sd":(0,0.3), "λ":(0.5,10.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0.,0.1)}

model_names = ["LPCHM", "EWA"]
param_names = dict()
param_names["EWA"] = ["p", "λ", "φ", "ρ", "δ"]
param_names["LPCHM"] = ["τ", "λ", "β"]

test_priors = [Distribution(**{key: RV("uniform", a, b - a) for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]



mturk_games = dict()
mturk_games[1] = [np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]]),np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]])]
mturk_games[2] = [np.array([[9.,0.],[0.,3.],[6.,6.]]), np.array([[3.,0.,0.],[0.,9.,0.]])]
mturk_games[3] = [np.array([[2.,2.,4.,4.],[8.,8.,2.,2.],[0.,2.,0.,2.],[6.,0.,6.,0.]]), np.array([[2.,4.,2.,4.],[2.,4.,8.,2.],[8.,2.,2.,4.],[8.,2.,8.,2.]])]
mturk_games[4] = [np.array([[2.,2.,2.,2.,2.],[1.,4.,4.,4.,4.],[1.,3.,10.,10.,10.],[1.,3.,5.,18.,18.],[1.,3.,5.,7.,30.]]), np.array([[0.,3.,3.,3.,3.],[0.,1.,7.,7.,7.],[0.,1.,4.,13.,13.],[0.,1.,4.,6.,23.],[0.,1.,4.,6.,8.]])]
mturk_games[5] = [np.array([[12.,4.,0.],[4.,12.,0.],[0.,14.,2.],[6.,6.,6.]]), np.array([[12.,4.,0.,0.], [4.,12.,0.,0.],[14.,0.,2.,0.]])]

params = test_priors[1].rvs()

#%%

games = mturk_games
gids = [1,2,3,4,5]
p1_size = 20
p2_size = 20
rounds = 29
n_runs = 1
gid = 1
default_init = [1., 2.]

res = EWA_osap(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size)

period = 1
pid = 0
p_role = 0
opp_role = (p_role + 1) % 2


arg_params = [params["p"], params["λ"], params["φ"], params["ρ"], params["δ"]]

self_strats = res[gid]["ind_hists"][p_role][pid,period,:]
self_payoff_mat = games[gid][p_role]
self_As = np.ones(len(games[gid][p_role]))
self_N = 1.
match = int(res[gid]["matches"][p_role][period][pid])
opp_s = res[gid]["ind_hists"][opp_role][match,period,:]
matches =  np.array([0])
params = np.array(arg_params)

self_strats_old = copy.deepcopy(self_strats)

osap_EWA_pred(self_strats, self_payoff_mat, self_As, self_N, opp_s, params)

self_strats
self_strats_old

self_strats = np.array([res[gid]["ind_hists"][p_role][pid,period,:]])
self_payoff_mat = games[gid][p_role]
self_As = np.array([np.ones(len(games[gid][p_role]))])
self_Ns = np.array([1.])
opp_s = np.array([res[gid]["ind_hists"][opp_role][int(res[gid]["matches"][p_role][period][pid]),period,:]])
matches =  np.array([0])
params = np.array(arg_params)
EWA_reply_for(self_strats, self_payoff_mat, self_As, self_Ns, opp_s, matches, arg_params)


len(self_payoff_mat)

self_payoff_mat[0].shape[0]
self_strats = [res[gid]["ind_hists"][p_role][pid,period,:]]
self_payoff_mat = games[gid][p_role]
self_As = [np.ones(len(games[gid][p_role]))]
self_Ns = [1.]
opp_s = [res[gid]["ind_hists"][opp_role][int(res[gid]["matches"][p_role][period][pid]),period,:]]
matches =  [0]
params = arg_params

n_strats = self_payoff_mat.shape[0]
for i in range(len(self_strats)):
i = 0
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
self_strats
for s in range(n_strats):
    s = 1
    self_As[i][s] = (φ*old_N*self_As[i][s] + δ*avg_payoffs[s] + (1-δ)*self_strats[i][s]*prev_payoff)/self_Ns[i]
    if p > np.random.rand():
        new_strats = np.exp(λ*self_As[i] - (λ*self_As[i]).max())
        new_strats = new_strats/new_strats.sum()
        reply = np.zeros(n_strats)
        reply[weighted_rand_int(new_strats)] = 1.
        self_strats[i][:] = reply[:]

res[gid]["ind_hists"][opp_role][:,period,:]
res[gid]["matches"][p_role][pid]

dict(params)
