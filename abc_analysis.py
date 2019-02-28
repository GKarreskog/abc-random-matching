
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
# n_runs = 1
# bw = 0.05
# n_particles = 100
# max_pops = 5
# min_accept_rate = 0.01

# init_ε = 3.
# α = 0.5

n_per_model = 1

def LPCHM_wrap(params, random_params=True):
    return LPCHM_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)

def LBR_wrap(params, random_params=True):
    return LBR_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)

def EWA_wrap(params, random_params=True):
    return EWA_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)

def gid_wrap_model(model, gid, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True):
    def call(params):
        result = model(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)
        return result
    return call

    # return(model(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params))


# model_names = ["LBR", "LPCHM", "EWA"]
# models = [LBR_wrap, LPCHM_wrap, EWA_wrap]
# models_for_gid = [LBR_model, LPCHM_model, EWA_model]
# model_names = ["LBR", "EWA"]
# models_wrap = [LBR_wrap, EWA_wrap]
# models = [LBR_model, EWA_model]
model_names = ["EWA"]
models_wrap = [EWA_wrap]
models = [EWA_model]
# models_for_gid = [LBR_model, EWA_model]

gid_models = {gid: [gid_wrap_model(model, gid, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True) for model in models_for_gid] for gid in gids}

# osap_models = [LBR_osap, LPCHM_osap, EWA_osap]
osap_models = [LBR_osap, EWA_osap]

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

default_init = [1., 2.]





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
models = []
models_wrap = []
osap_models = []
for name, restriction in zip(["EWA_0", "EWA_05", "EWA_1"], [{"p":0.99, "δ":0.}, {"p":0.99, "δ":0.5}, {"p":0.99, "δ":1.}]):
    wrap_fun, osap_model, perf_f_restricted, param_names_restricted, param_space_restricted, sample_param_space_restricted, bounds_restricted = gen_restricted_ewa({"δ":0.}, EWA_model, EWA_osap, param_names["EWA"], param_spaces["EWA"], sample_param_spaces["EWA"], perf_EWA, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=False)
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


#%%
both_df_sse = pd.read_pickle("both_df_sse.pkl")
#%% Individual ABC, ll compare between models


n_tests = 1
for j in range(n_tests):
    for i in range(len(model_names)):
        print("----- ", j+1, "/", n_tests, ", model ", model_names[i], "--------")
        res_vec = test_both_abc_osap(i, model_names, perf_fs, osap_models, models_wrap, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict({"max_pops":10, "n_particles":1000, "min_acceptance_rate":0.05, "init_ε":8.5, "α":0.4}))
        both_df_sse = both_df_sse.append(res_vec, ignore_index=True)
        both_df_sse.to_pickle("both_df_sse.pkl")
        print(both_df_sse)
        print(tabulate(both_df_sse, headers='keys', tablefmt='psql'))
        # print(both_df_sse.append(res_vec, ignore_index=True))

both_df_sse


#%%
both_df = pd.read_pickle("both_df.pkl")
#%% Individual ABC, ll compare between models
n_tests = 1
for j in range(n_tests):
    for i in range(len(models)):
        print("----- ", j+1, "/", n_tests, "--------")
        res_vec = gen_abc_ll_osap_model_params_test(i, model_names, models, perf_fs, osap_models, gid_models, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, 100000, options=dict({"max_pops":7, "n_particles":1000, "min_acceptance_rate":0.05, "init_ε":3., "α":0.5}))
        both_df = both_df.append(res_vec, ignore_index=True)
        both_df.to_pickle("both_df.pkl")
        print(both_df)
        print(tabulate(both_df, headers='keys', tablefmt='psql'))
        # print(both_df.append(res_vec, ignore_index=True))

ind_obs_df, var_bias_df = gen_params_both_dfs(model_names, param_names, both_df)


#TODO: Testa PCA för att fastställa rimliga summary statistics för ABC. Kolla även på pappret för rimliga summary statistics.
#%%

groupded = both_df.groupby('id')
ll_type = "train_osap_osap"
# ll_type = "train_ll_osap_07"
ll_type = "test_ll_abc_07"
both_df[ll_type]
i = 0
correct = 0
for name, group in groupded:
    group = group.sort_values(by=[ll_type], ascending=False)
    # print(group.first)
    i += 1
    correct += group.iloc[0]["model"] == group.iloc[0]["true_model"]
    print(group[["id", "model", "true_model", ll_type]].iloc[0]["model"] == group.iloc[0]["true_model"])
    print(group[["id", "model", "true_model", ll_type]])
print(correct)
print(i)


#%% Run and compare estimatesn
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
models_df_sse = pd.read_pickle("models_df_sse.pkl")
params_df_sse = pd.read_pickle("params_df_sse.pkl")
# models_est_vec = []
#%%
# params_est_vec = []
n_tests = 1
for j in range(n_tests):
    for i in range(len(models)):
        print("----- ", j+1, "/", n_tests, "--------")
        model_test = gen_test_and_compare_models_abc_osap(i, model_names, models, perf_fs, osap_models, gid_models, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict({"max_pops":7, "n_particles":600, "min_acceptance_rate":0.05, "init_ε":init_ε}))
        # models_est_vec.append(model_test)
        print(model_test)
        # params_test = gen_test_and_compare_params_abc_osap(i, model_names, models, perf_fs, osap_models, gid_models, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, default_init, options=dict({"max_pops":10, "n_particles":300, "min_acceptance_rate":0.05, "init_ε":init_ε}))
        # params_est_vec.append(params_test)
        # print(params_test)
        models_df_sse = models_df_sse.append(model_test, ignore_index=True)
        models_df_sse.to_pickle("models_df_sse.pkl")
        params_df_sse.to_pickle("params_df_sse.pkl")
        print(models_df_sse)
        print(params_df_sse)
ind_obs_df, var_bias_df = gen_params_perf_dfs(model_names, param_names, params_df_sse)


#%%
def gen_restricted_ewa(restrictions, param_names, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=False):
    def wrap_fun(params):
        for param in restrictions.keys():
            params[param] = restrictions[param]
        res  = EWA_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)
        return res
    return wrap_fun




#%%
i
ii = 0
gid = 1
training_params = dict(test_priors[i].rvs())
training_res = osap_models[i](training_params, gids, games, default_init, rounds, p1_size, p2_size)
# test_res = osap_models[i](training_params, [gid], games, default_init, rounds, p1_size, p2_size)
dfs, ws, abc_mod_pred = abc_from_res(training_res, gids, model_names, gid_models, priors, param_spaces, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})

#
#
list(training_res[2].values())
flat_hists = training_res["flat_hists"]
training_res["shape"]
data = np.array([flatten_single_hist(training_res[2]["pop_hists"])])
flatten_single_hist(training_res[2]["pop_hists"])
shape = [data.shape]
shape = training_res["shape"]
simulated = unflatten_data(flat_hists, shape)
y = {"data": flat_hists, "shape":shape}
abc_hist = abc_from_data(y, model_names, gid_models[gid], priors, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})
abc_hist.get_model_probabilities().get_values()[abc_hist.max_t]
# mod_probs = abc_hist.get_model_probabilities()
# dist = RVmodel(abc_hist)
# dist.rvs()
#
# len(abc_hist.alive_models(abc_hist.max_t))
# dfs = dict()
# ws = dict()
# for m in mod_probs.columns:
#     dfs[m], ws[m] = abc_hist.get_distribution(m)
#
# priors = [priors_from_posterior(dfs[m], ws[m], param_spaces[model_names[m]]) for m in mod_probs.columns]
#
# gid = 2
# training_params = dict(test_priors[i].rvs())
# training_res = osap_models[i](training_params, [gid], games, default_init, rounds, p1_size, p2_size)
# # test_res = osap_models[i](training_params, [gid], games, default_init, rounds, p1_size, p2_size)
#
# flat_hists = training_res["flat_hists"]
# shape = training_res["shape"]
# simulated = unflatten_data(flat_hists, shape)
# y = {"data": flat_hists, "shape":shape}
# abc_hist = abc_from_data(y, model_names, gid_models[gid], priors, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})
# mod_probs = abc_hist.get_model_probabilities()
# dfs = dict()
# ws = dict()
# m_active = 0
# for m in abc_hist.alive_models(abc_hist.max_t):
#     dfs[m], ws[m] = abc_hist.get_distribution(m)
#
# model_names = [model_names[m] for m in abc_hist.alive_models(abc_hist.max_t)]
#
#
#
# priors = [priors_from_posterior(dfs[m], ws[m], param_spaces[model_names[m]]) for m in mod_probs.columns]

#%%
n_tests = 2
for j in range(n_tests):
    for i in range(len(models)):
        print("----- ", i, "/", n_tests, "--------")
        model_test = gen_test_and_compare_models_abc_osap(i, model_names, models, perf_fs, osap_models, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict({"max_pops":7, "n_particles":100}))
        models_est_vec.append(model_test)
        print(model_test)
        params_test = gen_test_and_compare_params_abc_osap(i, model_names, models, perf_fs, osap_models, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, default_init, options=dict({"max_pops":10, "n_particles":50}))
        params_est_vec.append(params_test)
        print(params_test)

models_est_vec
params_est_vec


#%%
i = 0
training_params = dict(test_priors[i].rvs())
training_res = osap_models[i](training_params, gids, games, default_init, rounds, p1_size, p2_size)

test_res = osap_models[i](training_params, gids, games, default_init, rounds, p1_size, p2_size)

flat_hists = training_res["flat_hists"]
shape = training_res["shape"]
simulated = unflatten_data(flat_hists, shape)
y = {"data": flat_hists, "shape":shape}

abc_hist = abc_from_data(y, model_names, models, prior, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})
abc_hist = abc_from_data(y, model_names, models, posterior, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})
abc_hist.get_model_probabilities()
df2, w2 = abc_hist.get_distribution(1)

posteriors = [Distribution(**{key: RVkde(df, w, key, a, b) for key, (a,b) in param_spaces["LPCHM"].items()}), Distribution(**{key: RVkde(df2, w2, key, a, b) for key, (a,b) in param_spaces["EWA"].items()})]


posteriors[0]
priors
abc_hist = abc_from_data(y, model_names, models, posteriors, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})

abc_hist.get_model_probabilities()
prior_β = RVkde(df,w, "λ", 0., 10.)
# prior_β.pdf(pd.DataFrame({"β":[0.1]}))

prior_β.rvs()
x = 0.3
kde_1d(df, w, "λ", xmin=None, xmax=None, numx=50)
x = pd.DataFrame({prior_β.key:[x]})
prior_β.kde.pdf(x)
prior_β.pdf(0.3)
prior_β.cdf(14.)
x = pd.DataFrame({"β":[0.1]})

prior_β.kde.pdf(pd.Series([0.432617]))
prior_β.kde.rvs()
model_probabilities = abc_hist.get_model_probabilities()
mod_probs = model_probabilities.get_values()[abc_hist.max_t]
mod_probs[i]

osap_pred = pred_model_and_params(training_res, perf_fs, model_names, bounds, games, gids)
perfs = [x["perf"] for x in osap_pred]

osap_est = osap_pred[i]["params"]
abc_est = ml_from_abc(df, w)
pop_perf(perf_fs[model_names[i]], training_res, games, gids, osap_est)
pop_perf(perf_fs[model_names[i]], training_res, games, gids, abc_est)

pop_perf(perf_fs[model_names[i]], test_res, games, gids, osap_est)
pop_perf(perf_fs[model_names[i]], test_res, games, gids, abc_est)

kde_data_abc = LPCHM_model(abc_est, gids, games, default_init, rounds, 1000, p1_size, p2_size, random_params=False)

kde_data_abc["shape"]
kde_hists = unflatten_data(kde_data_abc["data"], kde_data_abc["shape"])
test_hists = unflatten_data(test_res["flat_hists"], test_res["shape"])
kde_01 = gen_kde_from_flat(kde_hists[1], bw=0.1)
l001 = calc_from_kde(kde_01, test_hists[1])


res_unflattened = likelihood_from_sim(LPCHM_model, test_res, abc_est, gids, games, default_init, rounds, 10, p1_size, p2_size, cores=4)

res = gen_test_and_compare_models_abc_osap(1, model_names, models, perf_fs, osap_models, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict({"max_pops":2, "n_particles":100}))
res_params = gen_test_and_compare_params_abc_osap(1, model_names, models, perf_fs, osap_models, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, default_init, options=dict({"max_pops":2, "n_particles":100}))
# np.vstack([x[1] for x in res_unflattened]).shape
# len(res_unflattened)

ll_abc = likelihood_from_sim(LPCHM_model, test_res, abc_est, gids, games, default_init, rounds, 100, p1_size, p2_size)
ll_osap = likelihood_from_sim(LPCHM_model, test_res, osap_est, gids, games, default_init, rounds, 100, p1_size, p2_size)

osap_est
# test_res = osap_models[i](training_params, gids, games, default_init, rounds, p1_size, p2_size)
i = 0
np.ones(len(bounds[model_names[i]]))
#%%

print(models)
print(model_names)
n_correct = np.zeros(len(model_names))

for n in range(n_per_model):
    for i in range(len(models)):
        print("Starting n="+str(n))
        print(n_correct)
#         params = test_params[model_names[i]]
        params = test_priors[i].rvs()
        res = models[i](params)
        flat_hists = res["data"]
        shape = res["shape"]
        simulated = unflatten_data(flat_hists, shape)
        y = {"data": flat_hists, "shape":shape}
#         max_like = calc_likelihood(y["data"], y["data"])

#         print("SSE gen-Pseudo: " + str(max_like) + " - " + str(n_particles))
        abc = ABCSMC(models, priors, distance, population_size=n_particles, eps=pyabc.epsilon.QuantileEpsilon(initial_epsilon=init_ε, alpha=α))
        db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp-pseudos.db"))
        print(db_path)
        abc.new(db_path, y, meta_info={"bw":bw, "Data":"gen-Pseudo SSE", "distribution":"Trunc Normal"})
        history = abc.run(minimum_epsilon=0.1, max_nr_populations=max_pops,  min_acceptance_rate=min_accept_rate)
        model_probabilities = history.get_model_probabilities()
        print("Model is" + str(i))
#         print(params_to_μσ(params, model_names[i]))
#         print(params)
        print(model_probabilities)
        estim_dict[model_names[i]].append(model_probabilities.get_values()[history.max_t])
        if (model_probabilities.columns[model_probabilities.get_values()[history.max_t].argmax()] == i):
            print("correct!")
            n_correct[i] = n_correct[i] + 1
            df, w = history.get_distribution(m=i)
            estim_params = dict(df.multiply(w, axis=0).sum(0))
            print(pd.DataFrame([params, estim_params]))
#             print(estim_params)
            model_name = model_names[i]
            if (history.max_t+1 == max_pops):
                for param in param_names[model_name]:
                    observation = params[param]
                    fig, ax = plt.subplots()
                    for t in range(history.max_t+1):
                        df, w = history.get_distribution(m=i, t=t)
        #                 df_copy = df.copy()
        #                 a, b = αβ_lims[model_name][param]
        #                 df[param] = df_copy.apply(lambda x: st.beta.mean(x[param], x[param+"_sd"])*(b-a) + a, axis=1)
        # #                 df[param+"_sd"] = df_copy.apply(lambda x: st.beta.std(x[param], x[param+"_sd"])*(b-a), axis=1)
                        plot_kde_1d(df, w,
                                    xmin=param_spaces[model_name][param][0], xmax=param_spaces[model_name][param][1],
                                    x=param, ax=ax,
                                    label="PDF t={}".format(t))
                    ax.axvline(observation, color="k", linestyle="dashed");
                    ax.legend();
        #             plt.show()
#                     plt.savefig("fig/params/popgen6" + param + "-" + str(n_particles) + "-" + str(max_pops) + "-" + model_names[i] + "-" + str(n)+".png")
                    plt.show()
        else:
            print("incorrect!")
            df, w = history.get_distribution(m=model_probabilities.get_values()[history.max_t].argmax())
            estim_params_best = dict(df.multiply(w, axis=0).sum(0))
            df, w = history.get_distribution(m=i)
            estim_params_correct = dict(df.multiply(w, axis=0).sum(0))
            print(pd.DataFrame([params, estim_params_correct, estim_params_best]))

#         file.write("Model" + model_names[i] +" n=" + str(n) + " correct:" + str(n_correct[0]) + ", " + str(n_correct[1]))

#%%
i = 1
params = test_priors[i].rvs()
res = models[i](params)
flat_hists = res["data"]
shape = res["shape"]

simulated = unflatten_data(flat_hists, shape)
y = {"data": flat_hists, "shape":shape}
abc_from_data(y, model_names, models, priors, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})

#%%

def priors_from_kde(df,w):
    prior_dict = {}
    for key in df.columns:
        kde = MultivariateNormalTransition(scaling=1)
        kde.fit(df[[key]], w)
        x = kde.rvs(1000)
        α,β,loc,scale = scst.beta.fit(x[key], loc=0)
        prior_dict.update({key: RV("beta", α,β,loc,scale)})
    return(Distribution(**prior_dict))


def ml_from_abc(df, w):
    params = {}
    for key in df.columns:
        x, pdf = kde_1d(df, w, key)
        i = np.argmax(pdf)
        params[key] = x[i]
    return params



i = 0
test_params = test_priors[i].rvs()
priors = [Distribution(**{key: RV("uniform", a, b - a)
                for key, (a,b) in param_spaces[mod].items()}) for mod in model_names]



for gid in gids:

gid = 1
def LPCHM_wrap(params):
    return LPCHM_model(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size)

def EWA_wrap(params):
    return EWA_model(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size)

model_names = ["LPCHM", "EWA"]
models = [LPCHM_wrap, EWA_wrap]

res = models[i](test_params)
flat_hists = res["data"]
shape = res["shape"]
simulated = unflatten_data(flat_hists, shape)
y = {"data": flat_hists, "shape":shape}
history = abc_from_data(y, model_names, models, priors, n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})

history_0 = abc_from_data(y, [model_names[0]], [models[0]], [priors[0]], n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})
history_1 = abc_from_data(y, [model_names[1]], [models[1]], [priors[1]], n_particles, init_ε, α, max_pops, min_accept_rate, {"model":model_names[i]})

df0, w0 = history_0.get_distribution(m=0)
df1, w1 = history_1.get_distribution(m=0)

priors = [priors_from_kde(df0, w0), priors_from_kde(df1,w1)]

priors[1]["δ_sd"].rvs(10)

model_name = model_names[i]
if (model_probabilities.columns[model_probabilities.get_values()[history.max_t].argmax()] == i):
    for param in param_names[model_name]:
        observation = params[param]
        fig, ax = plt.subplots()
        for t in range(history.max_t+1):
            df, w = history.get_distribution(m=i, t=t)
#                 df_copy = df.copy()
#                 a, b = αβ_lims[model_name][param]
#                 df[param] = df_copy.apply(lambda x: st.beta.mean(x[param], x[param+"_sd"])*(b-a) + a, axis=1)
# #                 df[param+"_sd"] = df_copy.apply(lambda x: st.beta.std(x[param], x[param+"_sd"])*(b-a), axis=1)
            plot_kde_1d(df, w,
                        xmin=param_spaces[model_name][param][0], xmax=param_spaces[model_name][param][1],
                        x=param, ax=ax,
                        label="PDF t={}".format(t))
        ax.axvline(observation, color="k", linestyle="dashed");
        ax.legend();
#             plt.show()
#                     plt.savefig("fig/params/popgen6" + param + "-" + str(n_particles) + "-" + str(max_pops) + "-" + model_names[i] + "-" + str(n)+".png")
        plt.show()




#%% Testing OSAP
params = test_priors[1].rvs()
res = EWA_osap(params, [gid], games, default_init, rounds, n_runs, p1_size, p2_size)

res[1]["pop_hists"][1]
[np.mean(res[1]["ind_hists"][1][:,i,:],axis=0) for i in range(len(res[1]["ind_hists"][1][0,:,0]))]
#%%

df, w = history.get_distribution(m=1)
params = test_priors[1]
priors[1].decorator_repr()

pyabc.parameters.ParameterStructure

pyabc.parameters.from_dictionary_of_dictionaries()


#%%

from pyabc.visualization import kde_1d
from pyabc.transition import MultivariateNormalTransition
import pyabc.random_variables as rv
kde = MultivariateNormalTransition(scaling=1)
kde.fit(df[["p"]], w)

scst.beta.fit(kde.rvs(100), loc=0)

x, pdf = kde_1d(df, w, "p")

class PriorRV(rv.RVBase):
    def __init__

priors[1]["p"]


sckde = scst.gaussian_kde(df["p"])
sckde.pdf(0.9)


beta_priors = [Distribution(**{key: RV("beta", 0.4, 0.4, loc=a, scale=(b - a)) for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]

beta_priors[1]["p"].rvs()


def priors_from_kde(df,w):
    prior_dict = {}
    for key in df.columns:
        kde = MultivariateNormalTransition(scaling=1)
        kde.fit(df[[key]], w)
        x = kde.rvs(1000)
        α,β,loc,scale = scst.beta.fit(x[key])
        prior_dict.update({key: RV("beta", α,β,loc,scale)})
    return(Distribution(**prior_dict))


test_priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]
posteriors = priors_from_kde(df, w)

for key in df.columns:
    print("---Posterior for ", key, "-----")

    plt.plot(np.linspace(0.,1.,100), posteriors[key].pdf(np.linspace(0.,1.,100)))
    plt.show()

    print("---Actual kde for ", key, "-----")
    plot_kde_1d(df,w,key)
    plt.show()


key = "p"
a = 0
b = 1
kde = MultivariateNormalTransition(scaling=1)
kde.fit(df[[key]], w)
x = kde.rvs(1000)
x
α,β,a,b = scst.beta.fit(list(x[key]), loc=0)

posteriors["p"].pdf([0.4,0.3])

sckde = scst.gaussian_kde(df["p"])
x = sckde.resample(1000)
α, β, a,b = scst.beta.fit(x)

# scst.beta.fit(x[key])
dist = scst.beta(α,β,loc=a,scale=b)
plt.plot(np.linspace(0.,1.,100), dist.pdf(np.linspace(0.,1.,100)))
plt.show()

plot_kde_1d(df,w,"p")
plt.show()

df["p"]
