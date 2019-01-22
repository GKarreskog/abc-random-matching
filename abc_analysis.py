#%% Load functions
import randomMatching
from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap
from importlib import reload
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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
#%%
reload(randomMatching)
from randomMatching import LPCHM_model, EWA_model, distance, abc_from_data, EWA_osap

#%% Some setup
mturk_games = dict()
mturk_games[1] = [np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]]),np.array([[10.,0.,11.,0.],[12.,10.,5.,0.],[0.,12.,10.,0.],[18.,0.,0.,8.]])]
mturk_games[2] = [np.array([[9.,0.],[0.,3.],[6.,6.]]), np.array([[3.,0.,0.],[0.,9.,0.]])]
mturk_games[3] = [np.array([[2.,2.,4.,4.],[8.,8.,2.,2.],[0.,2.,0.,2.],[6.,0.,6.,0.]]), np.array([[2.,4.,2.,4.],[2.,4.,8.,2.],[8.,2.,2.,4.],[8.,2.,8.,2.]])]
mturk_games[4] = [np.array([[2.,2.,2.,2.,2.],[1.,4.,4.,4.,4.],[1.,3.,10.,10.,10.],[1.,3.,5.,18.,18.],[1.,3.,5.,7.,30.]]), np.array([[0.,3.,3.,3.,3.],[0.,1.,7.,7.,7.],[0.,1.,4.,13.,13.],[0.,1.,4.,6.,23.],[0.,1.,4.,6.,8.]])]
mturk_games[5] = [np.array([[12.,4.,0.],[4.,12.,0.],[0.,14.,2.],[6.,6.,6.]]), np.array([[12.,4.,0.,0.], [4.,12.,0.,0.],[14.,0.,2.,0.]])]

games = mturk_games
gids = [1,2,3,4,5]
p1_size = 20
p2_size = 20
rounds = 29
n_runs = 1
bw = 0.05
n_particles = 200
max_pops = 4
min_accept_rate = 0.01

init_ε = 10
α = 0.5

n_per_model = 1

estim_dict = dict()
for model in model_names:
    estim_dict[model] = []


param_spaces = dict()
param_spaces["LPCHM"] = {"τ":(0., 2.), "τ_sd":(0,0.3), "λ":(0.,10.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0., 0.2)}
param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,1.), "p":(0., 1), "p_sd":(0,0.2), "φ":(0, 1), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}

sample_param_spaces = dict()
sample_param_spaces["EWA"] = {"λ":(1.,8.), "λ_sd":(0,1.), "p":(0.3, 0.9), "p_sd":(0,0.2), "φ":(0., 1.), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}
sample_param_spaces["LPCHM"] = {"τ":(0.2, 1.8), "τ_sd":(0,0.3), "λ":(0.5,10.), "λ_sd":(0,1.), "β":(0.,1.), "β_sd":(0.,0.1)}

default_init = [1., 2.]

def LPCHM_wrap(params):
    return LPCHM_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size)

def EWA_wrap(params):
    return EWA_model(params, gids, games, default_init, rounds, n_runs, p1_size, p2_size)


model_names = ["LPCHM", "EWA"]
models = [LPCHM_wrap, EWA_wrap]


param_names = dict()
param_names["EWA"] = ["p", "λ", "φ", "ρ", "δ"]
param_names["LPCHM"] = ["τ", "λ", "β"]

priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in param_spaces[mod].items()}) for mod in model_names]

test_priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in sample_param_spaces[mod].items()}) for mod in model_names]



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
