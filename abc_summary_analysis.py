
#%% Load functions
import randomMatching
from randomMatching import *
import OSAP
from OSAP import *
from importlib import reload
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
# ABC stuff
from pyabc.visualization import plot_kde_matrix
from pyabc.visualization import plot_kde_1d, kde_1d
from pyabc.transition import MultivariateNormalTransition
import pyabc.random_variables as rv
import numpy as np
from RVkde import RVkde, priors_from_posterior

import multiprocessing as mp
import pickle
import os
import tempfile

import scipy.stats as st
import scipy as scp


from pyabc import (ABCSMC, RV, RVBase, Distribution,
                   PercentileDistanceFunction)

import pyabc
#%%
reload(randomMatching)
from randomMatching import *
reload(OSAP)
from OSAP import *
import RVkde
reload(RVkde)
from RVkde import RVkde, RVmodel, priors_from_posterior
#%% Some setup
mturk_games = dict()
mturk_games[1] = [np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]]),np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]])]
mturk_games[2] = [np.array([[9.,0.],[0.,3.],[6.,6.]]), np.array([[3.,0.,0.],[0.,9.,0.]])]
mturk_games[3] = [np.array([[2.,2.,4.,4.],[8.,8.,2.,2.],[0.,2.,0.,2.],[6.,0.,6.,0.]]), np.array([[2.,4.,2.,4.],[2.,4.,8.,2.],[8.,2.,2.,4.],[8.,2.,8.,2.]])]
mturk_games[4] = [np.array([[2.,2.,2.,2.,2.],[1.,4.,4.,4.,4.],[1.,3.,10.,10.,10.],[1.,3.,5.,18.,18.],[1.,3.,5.,7.,30.]]), np.array([[0.,3.,3.,3.,3.],[0.,1.,7.,7.,7.],[0.,1.,4.,13.,13.],[0.,1.,4.,6.,23.],[0.,1.,4.,6.,8.]])]
mturk_games[5] = [np.array([[12.,4.,0.],[4.,12.,0.],[0.,14.,2.],[6.,6.,6.]]), np.array([[12.,4.,0.,0.], [4.,12.,0.,0.],[14.,0.,2.,0.]])]

sse_games = dict()
sse_games[1] = [np.array([[3.,0.], [0.,1.],[2.,2.]]), np.array([[1.,0.,0.], [0.,3.,0.]])]
sse_games[2] = [np.array([[3.,4.,3.,4.], [4.,3.,4.,3.]]), np.array([[5.,0.], [4.,3.], [3.,4.], [0.,5.]])]
sse_games[3] = [np.array([[6.,1.,2.], [5.,2.,3.], [8.,0.,3.]]), np.array([[6.,1.,2.], [5.,2.,3.], [8.,0.,3.]])]
sse_games[4] = [np.array([[2.,4.,4.,4.], [0.,3.,5.,5.], [0.,1.,4.,6.], [0.,1.,2.,5.]]), np.array([[2.,4.,4.,4.], [0.,3.,5.,5.], [0.,1.,4.,6.], [0.,1.,2.,5.]])]
sse_games[5] = [np.array([[2.,2.,2.,2.],[2.,4.,0.,2.], [2.,0.,4.,2.], [2.,2.,2.,2.]]), np.array([[2.,2.,2.,2.],[2.,4.,0.,2.], [2.,0.,4.,2.], [2.,2.,2.,2.]])]

games = sse_games
gids = [1,3,4,5]
# gids = [1]
default_init = [1., 1.5]
p1_size = 20
p2_size = 20
rounds = 29
summary = avg_and_divided_summary
# bw = 0.05
# n_particles = 100
# max_pops = 5
# min_accept_rate = 0.01

# init_ε = 3.
# α = 0.5

n_per_model = 1

def LPCHM_wrap(params, random_params=True):
    return {"summary":summary(LPCHM_osap(params, gids, games, default_init, rounds, p1_size, p2_size, random_params=random_params))}

def LBR_wrap(params, random_params=True):
    return {"summary":summary(LBR_osap(params, gids, games, default_init, rounds, p1_size, p2_size, random_params=random_params))}

def EWA_wrap(params, random_params=True):
    return {"summary":summary(EWA_osap(params, gids, games, default_init, rounds, p1_size, p2_size, random_params=random_params))}


    # return(model(params, [gid], games, default_init, rounds, p1_size, p2_size, random_params=random_params))


# model_names = ["LBR", "LPCHM", "EWA"]
# models = [LBR_wrap, LPCHM_wrap, EWA_wrap]
# models_for_gid = [LBR_model, LPCHM_model, EWA_model]
model_names = ["LBR", "EWA"]
models_wrap = [LBR_wrap, EWA_wrap]
models = [LBR_model, EWA_model]
osap_models = [LBR_osap, EWA_osap]
# model_names = ["EWA"]
# models_wrap = [EWA_wrap]
# models = [EWA_model]
# osap_models = [EWA_osap]

# osap_models = [LBR_osap, LPCHM_osap, EWA_osap]

estim_dict = dict()
for model in model_names:
    estim_dict[model] = []



param_spaces = dict()
param_spaces["LPCHM"] = {"τ":(0., 3.), "τ_sd":(0,0.3), "λ":(0.,4.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.2)}
param_spaces["LBR"] = {"p":(0.2, 1.), "p_sd":(0,0.3), "λ":(0.,4.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.4)}
param_spaces["EWA"] = {"λ":(0,4), "λ_sd":(0,1.), "p":(0.99, 1.), "p_sd":(0,0.01), "φ":(0, 1), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}
# param_spaces["LPCHM"] = {"init_τ":(0.,2.), "init_λ":(0.,3.), "τ":(0., 3.), "τ_sd":(0,0.3), "λ":(0.,4.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.2)}
# param_spaces["LBR"] = {"init_τ":(0.,2.), "init_λ":(0.,3.), "p":(0., 1.), "p_sd":(0,0.3), "λ":(0.,4.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.4)}
# param_spaces["EWA"] = {"init_τ":(0.,2.), "init_λ":(0.,3.), "λ":(0,4), "λ_sd":(0,1.), "p":(0.99, 1.), "p_sd":(0,0.01), "φ":(0, 1), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}

sample_param_spaces = dict()
sample_param_spaces["EWA"] = {"λ":(0.2,2.5), "λ_sd":(0.5,1.), "p":(0.99, 1.), "p_sd":(0.,0.01), "φ":(0., 1.), "φ_sd":(0.2,0.4), "ρ":(0,1), "ρ_sd":(0.2,0.4) , "δ":(0,1), "δ_sd":(0.2,0.5)}
sample_param_spaces["LPCHM"] = {"τ":(0.5, 2.), "τ_sd":(0.2,0.5), "λ":(0.5,3.), "λ_sd":(0.5,1.), "β":(0.,1.), "β_sd":(0.2,0.4)}
sample_param_spaces["LBR"] = {"p":(0.2, 0.9), "p_sd":(0.1,0.3), "λ":(0.5,3.), "λ_sd":(0.5,1.), "β":(0.5,1.), "β_sd":(0.2,0.4)}
# sample_param_spaces["EWA"] = {"init_τ":(0.3,1.7), "init_λ":(0.4,1.5), "λ":(0.5,3.), "λ_sd":(0.5,1.), "p":(0.99, 1.), "p_sd":(0.,0.01), "φ":(0., 1.), "φ_sd":(0.2,0.4), "ρ":(0,1), "ρ_sd":(0.2,0.4) , "δ":(0,1), "δ_sd":(0.2,0.5)}
# sample_param_spaces["LPCHM"] = {"init_τ":(0.3,1.7), "init_λ":(0.4,1.5), "τ":(0.5, 2.), "τ_sd":(0.2,0.5), "λ":(0.5,3.), "λ_sd":(0.5,1.), "β":(0.,1.), "β_sd":(0.2,0.4)}
# sample_param_spaces["LBR"] = {"init_τ":(0.3,1.7), "init_λ":(0.4,1.5), "p":(0.2, 0.9), "p_sd":(0.1,0.3), "λ":(0.5,3.), "λ_sd":(0.5,1.), "β":(0.5,1.), "β_sd":(0.2,0.4)}

bounds = dict()
bounds["EWA"] = [param_spaces["EWA"]["p"], param_spaces["EWA"]["λ"], param_spaces["EWA"]["φ"], param_spaces["EWA"]["ρ"], param_spaces["EWA"]["δ"]]
bounds["LPCHM"] = [param_spaces["LPCHM"]["τ"], param_spaces["LPCHM"]["λ"], param_spaces["LPCHM"]["β"]]
bounds["LBR"] = [param_spaces["LBR"]["p"], param_spaces["LBR"]["λ"], param_spaces["LBR"]["β"]]

perf_fs = dict()
perf_fs["EWA"] = perf_EWA
perf_fs["LPCHM"] = perf_LPCHM
perf_fs["LBR"] = perf_LBR





param_names = dict()
param_names["EWA"] = ["p", "λ", "φ", "ρ", "δ"]
param_names["LPCHM"] = ["τ", "λ", "β"]
param_names["LBR"] = ["p", "λ", "β"]



priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in param_spaces[mod].items()}) for mod in model_names]

test_priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]
#%%
model_names = []
# models = []
models_wrap = []
osap_models = []
# for name, restriction in zip(["EWA_0", "EWA_05", "EWA_1"], [{"p":0.99, "p_sd":0.01, "δ":0., "δ_sd":0.01}, {"p":0.99, "p_sd":0.01, "δ":0.5, "δ_sd":0.01}, {"p":0.99, "p_sd":0.01, "δ":1., "δ_sd":0.01}]):
for name, restriction in zip(["EWA_0", "EWA_1"], [{"p":0.99, "p_sd":0.01, "δ":0., "δ_sd":0.01}, {"p":0.99, "p_sd":0.01, "δ":1., "δ_sd":0.01}]):
    wrap_fun, osap_model, perf_f_restricted, param_names_restricted, param_space_restricted, sample_param_space_restricted, bounds_restricted = gen_restricted_ewa(restriction, EWA_osap, param_names["EWA"], param_spaces["EWA"], sample_param_spaces["EWA"], perf_EWA, gids, games, default_init, rounds, p1_size, p2_size, random_params=False, summary=summary)
    model_names.append(name)
    models_wrap.append(wrap_fun)
    osap_models.append(osap_model)
    perf_fs[name] = perf_f_restricted
    param_names[name] = param_names_restricted
    param_spaces[name] = param_space_restricted
    sample_param_spaces[name] = sample_param_space_restricted
    bounds[name] = bounds_restricted



priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in param_spaces[mod].items()}) for mod in model_names]

test_priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]

#%% Individual ABC, ll compare between models
i = 1
train_params = dict(test_priors[i].rvs())
x = osap_models[i](train_params, gids, games, default_init, rounds, p1_size, p2_size)
y = osap_models[i](train_params, gids, games, default_init, rounds, p1_size, p2_size)

dfs, ws, best_model, end_mod_probs = abc_from_res(y, model_names, models_wrap, priors, 100, 0.5, 0.3, 7, 0.001, {"dist":"summary", "model":model_names[i]}, summary=summary)

for param in param_names[model_names[i]]:
    print(param)
    print(train_params[param])
    print(ml_from_abc(dfs[i],ws[i])[param])
    print(mean_from_abc(dfs[i],ws[i])[param])


#%%
both_df_sse_summary = pd.read_pickle("both_df_sse_summary.pkl")
#%% Individual ABC, ll compare between models
n_tests = 15
for j in range(n_tests):
    for i in range(len(model_names)):
        print("----- ", j+1, "/", n_tests, ", model ", model_names[i], "--------")
        res_vec = test_both_abc_osap(i, model_names, perf_fs, osap_models, models_wrap, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict({"max_pops":7, "n_particles":100, "min_acceptance_rate":0.001, "min_epsilon":0.15, "init_ε":0.5, "α":0.4, "stop_single":False}))
        both_df_sse_summary = both_df_sse_summary.append(res_vec, ignore_index=True)
        both_df_sse_summary.to_pickle("both_df_sse_summary.pkl")
        print(both_df_sse_summary)
        print(tabulate(both_df_sse_summary, headers='keys', tablefmt='psql'))
        # print(both_df_sse_summary.append(res_vec, ignore_index=True))

both_df_sse_summary
#%%
# ind_obs_df, var_bias_df = gen_params_both_dfs(model_names, param_names, both_df_sse_summary)
ind_obs_df, var_bias_df = gen_params_perf_dfs(model_names, param_names, both_df_sse_summary)


#%%
i = 1
train_params = dict(test_priors[i].rvs())
y = osap_models[i](train_params, gids, games, default_init, rounds, p1_size, p2_size)
y_sum = {"summary": summary(y)}
ε_levels = [0.05,0.1,0.13,0.15,0.17,0.2,0.25,0.3]
param_ε_vec = []
n_sims = 1000
for _ in range(10):
    ε_dict = {ε:0 for ε in ε_levels}
    train_params = dict(test_priors[i].rvs())
    y = osap_models[i](train_params, gids, games, default_init, rounds, p1_size, p2_size)
    y_sum = {"summary": summary(y)}
    for s in range(n_sims):
        x = osap_models[i](train_params, gids, games, default_init, rounds, p1_size, p2_size)
        x_sum = {"summary": summary(x)}
        dist_sum = distance_summary(y_sum, x_sum)
        for ε in ε_levels:
            if  dist_sum < ε:
                ε_dict[ε] += 1
    param_ε_vec.append({"params":train_params, "ε_dict":copy.deepcopy(ε_dict)})

param_ε_vec

#%%
train_params = dict(test_priors[0].rvs())
gid = 1
games[gid]
train_params
i = 2
plot_h(osap_models[i](train_params, [gid], games, default_init, rounds, p1_size, p2_size)[gid]["pop_hists"])

#
# def avg_play_summary(res):
#     sum = []
#     gids = list(filter(lambda k: k not in ["flat_hists", "shape"], x.keys()))
#     for gid in gids:
#         for role in [0,1]:
#          sum.extend(np.mean(res[gid]["pop_hists"][role], axis=0))
#     return np.array(sum)
#
# summary(x)
#
#
#
#
# def distance_summary(x,y):
#     gids = list(filter(lambda k: k not in ["flat_hists", "shape"], x.keys()))
#     tot_diff = np.sum(np.square(x["summary"] - y["summary"]))
#     return tot_diff
#
#
#
# def abc_from_data(y, model_names, models, priors, n_particles, init_ε, α, max_pops, min_accept_rate, add_meta_info, model_prior=None):
#     if model_prior == None:
#         model_prior = RV("randint", 0, len(model_names))
#     abc = ABCSMC(models, priors, distance_summary, model_prior=model_prior, population_size=n_particles, eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=init_ε, alpha=α))
#     db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp-pseudos.db"))
#     meta_info = {"distribution":"Trunc Normal"}
#     meta_info.update(add_meta_info)
#     abc.new(db_path, {"summary": summary(y)}, meta_info=meta_info)
#     history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_pops, min_accept_rate=min_accept_rate)
#     return(history)
