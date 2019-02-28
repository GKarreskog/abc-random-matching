#%% Load functions
import randomMatching
# from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap, EWA_reply_for, weighted_rand_int, poisson_weight, poisson_p
from randomMatching import *
from importlib import reload
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import dill as pickle
# import multiprocessing as mp
import pathos
import pathos.multiprocessing as mp
from pathos.pools import ProcessPool

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
# from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap, EWA_reply_for, weighted_rand_int, poisson_weight, poisson_p
from randomMatching import *

#%% OSAP functions
@jit(nopython=True)
def loglikelihood(pred, actual):
    ll = 0
    for s in range(len(pred)):
        ll += np.log(pred[s])*actual[s]
    return(ll)

def opt_ind_params(perf_f, res, games, gids, role, pid, init_params, bounds):
    def min_f(params_test):
        return(-perf_f(params_test, res, games, gids, role, pid))
    pool = ProcessPool(nodes=mp.cpu_count())
    opt = scp.optimize.differential_evolution(min_f, bounds, workers=pool.map)
    return opt
    pool.close()
    pool.join()
    pool.clear()

def opt_pop_params(perf_f, res, games, gids, init_params, bounds):
    def min_f(params):
        n_p1 = len(res[gids[0]]["params"][0])
        n_p2 = len(res[gids[0]]["params"][1])
        ll = 0
        for pid in range(n_p1):
            ll += perf_f(params, res, games, gids, 0, pid)
        for pid in range(n_p2):
            ll += perf_f(params, res, games, gids, 1, pid)
        return -ll
    pool = ProcessPool()
    opt = scp.optimize.differential_evolution(min_f, bounds, workers=pool.imap)
    pool.close()
    pool.join()
    pool.clear()
    # opt = scp.optimize.differential_evolution(min_f, bounds)
    return(opt)


def pop_perf(perf_f, res, games, gids, params):
    n_p1 = len(res[gids[0]]["params"][0])
    n_p2 = len(res[gids[0]]["params"][1])
    ll = 0
    for pid in range(n_p1):
        ll += perf_f(params, res, games, gids, 0, pid)
    for pid in range(n_p2):
        ll += perf_f(params, res, games, gids, 1, pid)
    return ll

###############################
########### LPCHM #############
@jit(nopython=True)
def osap_LPCHM_pred(self_strats, opp_guess, self_payoff_mat, opp_payoff_mat, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    τ = params[0]
    λ = params[1]
    β = params[2]
    opp_guess[:] = β*opp_guess[:] + opp_s[:]
    k_preds = np.zeros((4, n_strats))
    k_preds[0,:] = self_strats[:]
    for k in range(4):
        self_s_vec = np.ones((k+1, n_strats))/(n_strats)
        opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
        self_s_vec[0][:] = self_strats[:]
        # opp_strats_vec[0][:] = opp_s[match][:]
        opp_strats_vec[0][:] = opp_guess[:]/opp_guess.sum()
        poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
        for j in range(1, k+1):
            opp_s_guess = poisson_weight(poisson_weights_all, opp_strats_vec, j)
            self_s_guess = poisson_weight(poisson_weights_all, self_s_vec, j)
            if j < k:
                best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
            else:
                best_reply_logit(self_payoff_mat, opp_s_guess, λ, self_s_vec[j])
                best_reply_logit(opp_payoff_mat, self_s_guess, λ, opp_strats_vec[j])
        k_preds[k,:] = self_s_vec[-1][:]
    pred = np.zeros(n_strats)
    poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
    for k in range(4):
        pred[:] += poisson_weights_all[k]*k_preds[k,:]
    self_strats[:] = pred[:]/poisson_weights_all[0:4].sum()

# def perf_LPCHM(res, games, gids, role, pid, params):
def perf_LPCHM(params, res, games, gids, role, pid):
    opp_role = (role + 1) % 2
    n_periods = len(res[gids[0]]["matches"][role])
    if isinstance(params, dict):
        params = [params["τ"], params["λ"], params["β"]]
    for gid in gids:
            self_payoff_mat = games[gid][role]
            opp_payoff_mat = games[gid][opp_role]
            ll = 0
            opp_guess = np.ones(opp_payoff_mat.shape[0])/opp_payoff_mat.shape[0]
            for period in range(1,n_periods):
                match = int(res[gid]["matches"][role][period][pid])
                opp_s = res[gid]["ind_hists"][opp_role][match,period-1,:]
                pred_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period-1,:])
                actual_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period,:])
                osap_LPCHM_pred(pred_strats, opp_guess, self_payoff_mat, opp_payoff_mat, opp_s, params)
                ll += loglikelihood(pred_strats, actual_strats)
    return ll


###############################
########### LBR ###############
###############################
@jit(nopython=True)
def osap_LBR_pred(self_strats, opp_guess, self_payoff_mat, opp_payoff_mat, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    p = params[0]
    λ = params[1]
    β = params[2]
    opp_guess[:] = β*opp_guess[:] + opp_s[:]
    opp_play = opp_guess[:]/opp_guess.sum()
    lbr = np.zeros(n_strats)
    best_reply_logit(self_payoff_mat, opp_play, λ, lbr, pure=False)
    self_strats[:] = p*lbr[:] + (1-p)*self_strats[:]


# def perf_LBR(res, games, gids, role, pid, params):
def perf_LBR(params, res, games, gids, role, pid):
    opp_role = (role + 1) % 2
    n_periods = len(res[gids[0]]["matches"][role])
    if isinstance(params, dict):
        params = [params["p"], params["λ"], params["β"]]
    for gid in gids:
            self_payoff_mat = games[gid][role]
            opp_payoff_mat = games[gid][opp_role]
            ll = 0
            opp_guess = np.ones(opp_payoff_mat.shape[0])/opp_payoff_mat.shape[0]
            for period in range(1,n_periods):
                match = int(res[gid]["matches"][role][period][pid])
                opp_s = res[gid]["ind_hists"][opp_role][match,period-1,:]
                pred_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period-1,:])
                actual_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period,:])
                osap_LBR_pred(pred_strats, opp_guess, self_payoff_mat, opp_payoff_mat, opp_s, params)
                ll += loglikelihood(pred_strats, actual_strats)
    return ll



################################
############ EWA ###############
################################
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
    new_strats = np.exp(λ*self_As - (λ*self_As).max())
    new_strats = new_strats/new_strats.sum()
    self_strats[:] = (1-p)*self_strats[:] + p*new_strats[:]



def perf_EWA(params, res, games, gids, role, pid):
    opp_role = (role + 1) % 2
    if isinstance(params, dict):
        params = [params["p"], params["λ"], params["φ"], params["ρ"], params["δ"]]
    n_periods = len(res[gids[0]]["matches"][role])
    for gid in gids:
        self_payoff_mat = games[gid][role]
        self_As = self_payoff_mat.mean(axis=1)
        self_N = 1.
        ll = 0
        for period in range(1,n_periods):
            match = int(res[gid]["matches"][role][period][pid]) #TODO: Kolla så att match verkligen blir rätt
            opp_s = res[gid]["ind_hists"][opp_role][match,period-1,:]
            pred_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period-1,:])
            actual_strats = copy.deepcopy(res[gid]["ind_hists"][role][pid,period,:])
            osap_EWA_pred(pred_strats, self_payoff_mat, self_As, self_N, opp_s, params)
            ll += loglikelihood(pred_strats, actual_strats)
            self_N = params[3]*self_N + 1
    return ll

#### Generate restircted funs and spaces
def gen_restricted_ewa(restrictions, model, osap_model, param_names, param_space, sample_param_space, perf_f, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=True):
    for key,val in restrictions.items():
        restrictions[key] = np.float64(val)

    def wrap_fun(params_in, random_params=random_params):
        params = copy.deepcopy(params_in)
        if isinstance(params, list) or isinstance(params, np.ndarray):
            i = 0
            tot_params = []
            for p in param_names:
                if p in restrictions.keys():
                    tot_params.append(restrictions[p])
                else:
                    tot_params.append(params[i])
                i += 1
        else:
            for param in restrictions.keys():
                params[param] = restrictions[param]
                tot_params = params

        res  = model(tot_params, gids, games, default_init, rounds, n_runs, p1_size, p2_size, random_params=random_params)
        return res

    def osap_model_restricted(params_in, gids, games, default_init, rounds, p1_size, p2_size, random_params=random_params):
        params = copy.deepcopy(params_in)
        if isinstance(params, list) or isinstance(params, np.ndarray):
            i = 0
            tot_params = []
            for p in param_names:
                if p in restrictions.keys():
                    tot_params.append(restrictions[p])
                else:
                    tot_params.append(params[i])
                i += 1
        else:
            for param in restrictions.keys():
                params[param] = restrictions[param]
                tot_params = params
        res = osap_model(tot_params, gids, games, default_init, rounds, p1_size, p2_size, random_params=random_params)
        return res

    def perf_f_restricted(params_in, res, games, gids, role, pid):
        params = copy.deepcopy(params_in)
        if isinstance(params, list) or isinstance(params, np.ndarray):
            i = 0
            tot_params = []
            for p in param_names:
                if p in restrictions.keys():
                    tot_params.append(restrictions[p])
                else:
                    tot_params.append(params[i])
                i += 1
        else:
            for param in restrictions.keys():
                params[param] = restrictions[param]
                tot_params = params
        return perf_f(tot_params, res, games, gids, role, pid)

    param_names_restricted = [p for p in param_names if p not in restrictions.keys()]
    param_space_restricted = {k:v for (k,v) in param_space.items() if k not in restrictions.keys()}
    sample_param_space_restricted = {k:v for (k,v) in sample_param_space.items() if k not in restrictions.keys()}
    bounds_restricted = [param_space_restricted[p] for p in param_names_restricted]
    return (wrap_fun, osap_model_restricted, perf_f_restricted, param_names_restricted, param_space_restricted, sample_param_space_restricted, bounds_restricted)

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def pred_model_and_params(res, perf_fs, model_names, bounds, games, gids):
    res_vec = []
    for i in range(len(model_names)):
        params = np.ones(len(bounds[model_names[i]]))*0.5
        opt = opt_pop_params(perf_fs[model_names[i]], res, games, gids, params, bounds[model_names[i]])
        res_vec.append({"perf":-opt.fun, "params":opt.x})
    return res_vec

def gen_test_and_compare_models_abc_osap(mod_idx, model_names, models, perf_fs, osap_models, gid_models, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict(), random_params=True):
    opts = {"default_init": [1., 1.5], "init_ε":10., "α":0.5, "max_pops":10, "min_accept_rate":0.1, "n_particles":1000}
    for key,val in options.items():
        opts[key] = val
    i = mod_idx
    train_params = dict(test_priors[i].rvs())
    train_res = osap_models[i](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)

    # flat_hists = train_res["flat_hists"]
    # shape = train_res["shape"]
    # y = {"data": flat_hists, "shape":shape}
    # abc_hist = abc_from_data(y, model_names, models, priors, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})
    # model_probabilities = abc_hist.get_model_probabilities()
    # mod_probs = model_probabilities.get_values()[abc_hist.max_t]
    # abc_correct = (mod_probs.argmax() == i)
    dfs, ws, abc_mod_pred = abc_from_res_separate(train_res, gids, model_names, gid_models, priors, param_spaces, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})

    abc_correct = (abc_mod_pred == model_names[i])
    osap_pred = pred_model_and_params(train_res, perf_fs, model_names, bounds, games, gids)
    perfs = [x["perf"] for x in osap_pred]
    osap_correct = (np.argmax(perfs) == i)

    return dict({"model":model_names[i], "params":train_params, "abc_correct":abc_correct, "osap_correct":osap_correct})

def test_both_abc_osap(mod_idx, model_names, perf_fs, osap_models, models_wrap, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, options=dict()):
    opts = {"default_init": [1., 1.5], "init_ε":10., "α":0.5, "max_pops":10, "min_accept_rate":0.1, "n_particles":1000}
    for key,val in options.items():
        opts[key] = val
    i = mod_idx
    train_params = dict(test_priors[i].rvs())
    train_res = osap_models[i](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)

    # flat_hists = train_res["flat_hists"]
    # shape = train_res["shape"]
    # y = {"data": flat_hists, "shape":shape}
    # abc_hist = abc_from_data(y, model_names, models, priors, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})
    # model_probabilities = abc_hist.get_model_probabilities()
    # mod_probs = model_probabilities.get_values()[abc_hist.max_t]
    # abc_correct = (mod_probs.argmax() == i)
    dfs, ws, abc_mod_pred, end_mod_probs = abc_from_res(train_res, gids, model_names, models_wrap, priors, param_spaces, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})

    abc_correct = (abc_mod_pred == model_names[i])
    if i in dfs.keys():
        abc_est = ml_from_abc(dfs[i], ws[i])
        abc_mean_est = mean_from_abc(dfs[i], ws[i])
    else:
        abc_est = "Eliminated"

    osap_pred = pred_model_and_params(train_res, perf_fs, model_names, bounds, games, gids)
    osap_est = osap_pred[i]["params"]
    perfs = [x["perf"] for x in osap_pred]
    osap_correct = (np.argmax(perfs) == i)

    return dict({"model":model_names[i], "params":train_params, "abc_correct":abc_correct, "osap_correct":osap_correct, "abc_est":abc_est, "abc_mean_est":abc_mean_est, "osap_est":osap_est, "end_mod_probs":end_mod_probs})

def gen_test_and_compare_params_abc_osap(mod_idx, model_names, models, perf_fs, osap_models, gid_models_org, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, default_init, options=dict()):
    opts = {"default_init": [1., 1.5], "init_ε":10., "α":0.5, "max_pops":10, "min_accept_rate":0.1, "n_particles":500}
    gid_models = copy.deepcopy(gid_models_org)
    if len(options) > 0:
        for key,val in options.items():
            opts[key] = val
    i = mod_idx
    train_params = dict(test_priors[i].rvs())
    train_res = osap_models[i](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)
    test_res = osap_models[i](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)
    for gid in gids:
        gid_models[gid] = [gid_models[gid][i]]
    dfs, ws, abc_mod_pred = abc_from_res(train_res, gids, [model_names[i]], gid_models, [priors[i]], param_spaces, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})
    #
    # flat_hists = train_res["flat_hists"]
    # shape = train_res["shape"]
    # y = {"data": flat_hists, "shape":shape}
    # abc_hist = abc_from_data(y, model_names[i], models[i], priors[i], opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})
    #
    # df, w = abc_hist.get_distribution(0)
    # abc_est = ml_from_abc(dfs[max(gids)][0], ws[max(gids)][0])
    abc_est = median_from_abc(dfs[max(gids)][0], ws[max(gids)][0])


    osap_pred = pred_model_and_params(train_res, dict({model_names[i]: perf_fs[model_names[i]]}), [model_names[i]], dict({model_names[i]:bounds[model_names[i]]}), games, gids)
    osap_est = osap_pred[0]["params"]

    train_osap_osap = pop_perf(perf_fs[model_names[i]], train_res, games, gids, osap_est)
    train_osap_abc = pop_perf(perf_fs[model_names[i]], train_res, games, gids, abc_est)

    test_osap_osap = pop_perf(perf_fs[model_names[i]], test_res, games, gids, osap_est)
    test_osap_abc = pop_perf(perf_fs[model_names[i]], test_res, games, gids, abc_est)


    train_ll_abc = likelihood_from_sim(models[i], train_res, abc_est, gids, games, default_init, rounds, 10000, p1_size, p2_size, bw=0.05)
    train_ll_osap = likelihood_from_sim(models[i], train_res, osap_est, gids, games, default_init, rounds, 10000, p1_size, p2_size, bw=0.05)
    test_ll_abc = likelihood_from_sim(models[i], test_res, abc_est, gids, games, default_init, rounds, 10000, p1_size, p2_size,  bw=0.05)
    test_ll_osap = likelihood_from_sim(models[i], test_res, osap_est, gids, games, default_init, rounds, 10000, p1_size, p2_size, bw=0.05)


    return dict({"model":model_names[i], "params":train_params, "abc_est":abc_est, "osap_est":osap_est, "train_osap_osap":train_osap_osap, "train_osap_abc":train_osap_abc, "test_osap_osap":test_osap_osap, "test_osap_abc":test_osap_abc, "train_ll_abc":train_ll_abc, "train_ll_osap":train_ll_osap, "test_ll_abc":test_ll_abc, "test_ll_osap":test_ll_osap})


def gen_abc_ll_osap_model_params_test(mod_idx, model_names, models, perf_fs, osap_models, gid_models_org, param_spaces, priors, test_priors, bounds, games, gids, rounds, p1_size, p2_size, ll_ns=1000, options=dict()):
    opts = {"default_init": [1., 1.5], "init_ε":10., "α":0.5, "max_pops":10, "min_accept_rate":0.1, "n_particles":500}
    gid_models = copy.deepcopy(gid_models_org)
    if len(options) > 0:
        for key,val in options.items():
            opts[key] = val

    id = id_generator()
    dict_vec = []
    train_params = dict(test_priors[mod_idx].rvs())
    true_model = model_names[mod_idx]
    train_res = osap_models[mod_idx](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)
    test_res = osap_models[mod_idx](train_params, gids, games, opts["default_init"], rounds, p1_size, p2_size)
    for i in range(len(model_names)):
        gid_models = copy.deepcopy(gid_models_org)
        for gid in gids:
            gid_models[gid] = [gid_models[gid][i]]
        dfs, ws, abc_mod_pred = abc_from_res(train_res, gids, [model_names[i]], gid_models, [priors[i]], param_spaces, opts["n_particles"], opts["init_ε"], opts["α"], opts["max_pops"], opts["min_accept_rate"], {"model":model_names[i]})

        abc_est = ml_from_abc(dfs[max(gids)][0], ws[max(gids)][0])
        abc_mean_est = median_from_abc(dfs[max(gids)][0], ws[max(gids)][0])


        osap_pred = pred_model_and_params(train_res, dict({model_names[i]: perf_fs[model_names[i]]}), [model_names[i]], dict({model_names[i]:bounds[model_names[i]]}), games, gids)
        osap_est = osap_pred[0]["params"]
        print("OSAP done")

        train_osap_osap = pop_perf(perf_fs[model_names[i]], train_res, games, gids, osap_est)
        train_osap_abc = pop_perf(perf_fs[model_names[i]], train_res, games, gids, abc_est)

        test_osap_osap = pop_perf(perf_fs[model_names[i]], test_res, games, gids, osap_est)
        test_osap_abc = pop_perf(perf_fs[model_names[i]], test_res, games, gids, abc_est)

        print("Starting LL ABC estimation")
        # train_ll_abc_07, train_ll_abc_05, train_ll_abc_03, train_ll_abc_01 = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bws=[0.07, 0.05, 0.03, 0.01])
        # train_ll_osap_07, train_ll_osap_05, train_ll_osap_03, train_ll_osap_01 = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bws=[0.07, 0.05, 0.03, 0.01])
        # test_ll_abc_07, test_ll_abc_05, test_ll_abc_03, test_ll_abc_01 = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bws=[0.07, 0.05, 0.03, 0.01])
        # test_ll_osap_07, test_ll_osap_05, test_ll_osap_03, test_ll_osap_01 = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bws=[0.07, 0.05, 0.03, 0.01])

        abc_res = likelihood_from_sim(models[i], [train_res, test_res], abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bws=[0.07, 0.05, 0.03, 0.01], random_params=True)
        train_ll_abc_07, train_ll_abc_05, train_ll_abc_03, train_ll_abc_01 = abc_res[0]
        test_ll_abc_07, test_ll_abc_05, test_ll_abc_03, test_ll_abc_01 = abc_res[1]

        print("Starting LL OSAP estimation", sep=' ', end='n', file=sys.stdout, flush=False)
        osap_res = likelihood_from_sim(models[i], [train_res, test_res], osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bws=[0.07, 0.05, 0.03, 0.01])

        train_ll_osap_07, train_ll_osap_05, train_ll_osap_03, train_ll_osap_01 = osap_res[0]
        test_ll_osap_07, test_ll_osap_05, test_ll_osap_03, test_ll_osap_01  = osap_res[1]


        print("Finished LL estimation")
        # train_ll_abc_07 = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.07)
        # train_ll_osap_07 = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.07)
        # test_ll_abc_07 = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bw=0.07)
        # test_ll_osap_07 = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.07)

        # train_ll_abc_05 = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.05)
        # train_ll_osap_05 = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.05)
        # test_ll_abc_05 = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bw=0.05)
        # test_ll_osap_05 = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.05)

        #
        # train_ll_abc = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.05)
        # train_ll_osap = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size)
        #
        # test_ll_abc = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size)
        # test_ll_osap = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size)

        # train_ll_abc_03 = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.03)
        # train_ll_osap_03 = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.03)
        # test_ll_abc_03 = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bw=0.03)
        # test_ll_osap_03 = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.03)
        #
        # train_ll_abc_01 = likelihood_from_sim(models[i], train_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.01)
        # train_ll_osap_01 = likelihood_from_sim(models[i], train_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.01)
        # test_ll_abc_01 = likelihood_from_sim(models[i], test_res, abc_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size,  bw=0.01)
        # test_ll_osap_01 = likelihood_from_sim(models[i], test_res, osap_est, gids, games, opts["default_init"], rounds, int(ll_ns/64), p1_size, p2_size, bw=0.01)


        res_dict = dict({"true_model":true_model, "model":model_names[i], "id":id, "params":train_params, "abc_est":abc_est, "abc_mean_est":abc_mean_est, "osap_est":osap_est, "train_osap_osap":train_osap_osap, "train_osap_abc":train_osap_abc, "test_osap_osap":test_osap_osap, "test_osap_abc":test_osap_abc, "train_ll_abc_07":train_ll_abc_07, "train_ll_osap_07":train_ll_osap_07, "test_ll_abc_07":test_ll_abc_07, "test_ll_osap_07":test_ll_osap_07, "train_ll_abc_05":train_ll_abc_05, "train_ll_osap_05":train_ll_osap_05, "test_ll_abc_05":test_ll_abc_05, "test_ll_osap_05":test_ll_osap_05, "train_ll_abc_03":train_ll_abc_03, "train_ll_osap_03":train_ll_osap_03, "test_ll_abc_03":test_ll_abc_03, "test_ll_osap_03":test_ll_osap_03, "train_ll_abc_01":train_ll_abc_01, "train_ll_osap_01":train_ll_osap_01, "test_ll_abc_01":test_ll_abc_01, "test_ll_osap_01":test_ll_osap_01})
        dict_vec.append(res_dict)
        print(pd.DataFrame(dict_vec))
    return dict_vec




def gen_params_perf_dfs(model_names, param_names, params_df):
    ind_obs_vec = []
    var_bias_vec = []
    for i in range(len(model_names)):
        model = model_names[i]
        for param in param_names[model]:
            osap_bias = 0
            osap_var = 0
            abc_bias = 0
            abc_var = 0
            n = 0
            for index, row in params_df[params_df["model"] == model].iterrows():
                n += 1
                idx = param_names[model].index(param)
                abc_est = row["abc_est"][param]
                osap_est = row["osap_est"][idx]
                actual = row["params"][param]
                osap_bias += osap_est -actual
                osap_var += np.abs(actual - osap_est)
                abc_bias += abc_est - actual
                abc_var += np.abs(actual - abc_est)
                ind_obs_vec.append(dict({"model":model, "param":param, "actual":actual, "abc_est":abc_est, "osap_est":osap_est}))
            var_bias_vec.append(dict({"model":model, "param":param, "osap_bias":osap_bias/n, "abc_bias":abc_bias/n, "osap_var":osap_var/n, "abc_var":abc_var/n}))

    ind_obs_df = pd.DataFrame(ind_obs_vec)
    var_bias_df = pd.DataFrame(var_bias_vec)
    return (ind_obs_df, var_bias_df)

def gen_params_both_dfs(model_names, param_names, params_df):
    ind_obs_vec = []
    var_bias_vec = []
    for i in range(len(model_names)):
        model = model_names[i]
        for param in param_names[model]:
            osap_bias = 0
            osap_var = 0
            abc_bias = 0
            abc_var = 0
            abc_mean_bias = 0
            abc_mean_var = 0
            n = 0
            for index, row in params_df[(params_df["model"] == model) & (params_df["true_model"] == model)].iterrows():
                n += 1
                idx = param_names[model].index(param)
                abc_est = row["abc_est"][param]
                abc_mean_est = row["abc_mean_est"][param]
                osap_est = row["osap_est"][idx]
                actual = row["params"][param]
                osap_bias += osap_est -actual
                osap_var += np.abs(actual - osap_est)
                abc_bias += abc_est - actual
                abc_var += np.abs(actual - abc_est)
                abc_mean_bias += abc_mean_est - actual
                abc_mean_var += np.abs(actual - abc_mean_est)
                ind_obs_vec.append(dict({"model":model, "param":param, "actual":actual, "abc_est":abc_est, "abc_mean_est":abc_mean_est, "osap_est":osap_est}))
            var_bias_vec.append(dict({"model":model, "param":param, "osap_bias":osap_bias/n, "abc_bias":abc_bias/n,"abc_mean_bias":abc_mean_bias/n, "osap_var":osap_var/n, "abc_var":abc_var/n, "abc_mean_var":abc_mean_var/n}))

    ind_obs_df = pd.DataFrame(ind_obs_vec)
    var_bias_df = pd.DataFrame(var_bias_vec)
    return (ind_obs_df, var_bias_df)



#
# #%%
#
#
# param_spaces = dict()
# param_spaces["LPCHM"] = {"τ":(0.1, 2.), "τ_sd":(0,0.3), "λ":(0.,10.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.2)}
# param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,1.), "p":(0.1, 1), "p_sd":(0,0.2), "φ":(0, 1), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}
#
# sample_param_spaces = dict()
# sample_param_spaces["EWA"] = {"λ":(1.,8.), "λ_sd":(0,1.), "p":(0.3, 0.9), "p_sd":(0,0.2), "φ":(0., 1.), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}
# sample_param_spaces["LPCHM"] = {"τ":(0.2, 1.8), "τ_sd":(0,0.3), "λ":(0.5,10.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0.,0.1)}
#
# model_names = ["LPCHM", "EWA"]
# param_names = dict()
# param_names["EWA"] = ["p", "λ", "φ", "ρ", "δ"]
# param_names["LPCHM"] = ["τ", "λ", "β"]
#
# test_priors = [Distribution(**{key: RV("uniform", a, b - a) for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]
#
#
#
# mturk_games = dict()
# mturk_games[1] = [np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]]),np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]])]
# mturk_games[2] = [np.array([[9.,0.],[0.,3.],[6.,6.]]), np.array([[3.,0.,0.],[0.,9.,0.]])]
# mturk_games[3] = [np.array([[2.,2.,4.,4.],[8.,8.,2.,2.],[0.,2.,0.,2.],[6.,0.,6.,0.]]), np.array([[2.,4.,2.,4.],[2.,4.,8.,2.],[8.,2.,2.,4.],[8.,2.,8.,2.]])]
# mturk_games[4] = [np.array([[2.,2.,2.,2.,2.],[1.,4.,4.,4.,4.],[1.,3.,10.,10.,10.],[1.,3.,5.,18.,18.],[1.,3.,5.,7.,30.]]), np.array([[0.,3.,3.,3.,3.],[0.,1.,7.,7.,7.],[0.,1.,4.,13.,13.],[0.,1.,4.,6.,23.],[0.,1.,4.,6.,8.]])]
# mturk_games[5] = [np.array([[12.,4.,0.],[4.,12.,0.],[0.,14.,2.],[6.,6.,6.]]), np.array([[12.,4.,0.,0.], [4.,12.,0.,0.],[14.,0.,2.,0.]])]
#
# #%% Test LPCHM osap
# sim_params = test_priors[0].rvs()
# sim_params["λ"] = np.float64(1.4)
# sim_params = dict(sim_params)
#
#
# games = mturk_games
# gids = [1,2,3,4,5]
# p1_size = 20
# p2_size = 20
# rounds = 29
# gid = 1
# default_init = [1., 2.]
#
# res = LPCHM_osap(sim_params, gids, games, default_init, rounds, p1_size, p2_size)
# period = 1
# pid = 2
# p_role = 1
# opp_role = (p_role + 1) % 2
# bounds = [param_spaces["LPCHM"]["τ"], param_spaces["LPCHM"]["λ"], param_spaces["LPCHM"]["β"]]
# params = res[1]["params"][p_role][pid]
# perf_LPCHM(res, games, gids, p_role, pid, sim_params)
# params
# %timeit opt = opt_ind_params(perf_LPCHM, res, games, gids, p_role, pid, params, bounds)
# p, f = opt_ind_params(perf_LPCHM, res, games, gids, p_role, pid, params, bounds)
# %time opt = opt_pop_params(perf_LPCHM, res, games, gids, params, bounds)
# opt.fun
# perf_LPCHM(res, games, gids, p_role, pid, sim_params)
# perf_LPCHM(res, games, gids, p_role, pid, opt.x)
#
# params
# sim_params
# opt
# pop_perf(perf_LPCHM, res, games, gids, sim_params)
# pop_perf(perf_LPCHM, res, games, gids, opt.x)
# params
# sim_params
# opt.x
# params
# sim_params
# #%%
# sim_params = test_priors[1].rvs()
# games = mturk_games
# gids = [1,2,3,4,5]
# p1_size = 20
# p2_size = 20
# rounds = 29
# gid = 1
# default_init = [1., 2.]
#
# res = EWA_osap(sim_params, gids, games, default_init, rounds, p1_size, p2_size)
#
# period = 1
# pid = 2
# p_role = 1
# opp_role = (p_role + 1) % 2
#
# bounds_EWA = [param_spaces["EWA"]["p"], param_spaces["EWA"]["λ"], param_spaces["EWA"]["φ"], param_spaces["EWA"]["ρ"], param_spaces["EWA"]["δ"]]
# params = res[1]["params"][p_role][pid][:]
# params_EWA = test_priors[1].rvs()
# params[0] = 0.3
# params[1] = 0.2
#
#
# opt = opt_ind_params(perf_EWA, res, games, gids, p_role, pid, params_EWA, bounds)
# opt = opt_pop_params(perf_EWA, res, games, gids, params_EWA, bounds_EWA)
# opt
# perf_EWA(res, games, gids, p_role, pid, opt.x)
# perf_EWA(res, games, gids, p_role, pid, params)
# print(opt.x)
# print(sim_params)
# params_EWA
# pop_perf(perf_EWA, res, games, gids, dict(sim_params))
# pop_perf(perf_EWA, res, games, gids, opt.x)
# # pop_perf(perf_LPCHM, res, games, gids, sim_params)
#
# len(res[1]["params"][p_role])
#
#
