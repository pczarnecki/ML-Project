import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import scipy.constants as sp
import random
from sklearn.linear_model import LinearRegression
import itertools
from scipy.integrate import quad
from scipy.optimize import root_scalar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost
import joblib
from sklearn.model_selection import GridSearchCV

def compute_q_tilde(S, spectral_data, reference_data, num_hl, num_cols):
    q_v = spectral_data.isel(wavenumber = S).data
    S_original = S

    # do multiple linear regression over ensemble (Buehler), or least squares regression (Moncet)
    # require weights to be > 0 (discard wavenumbers where this is not true, indicates redundancy;
    # in case of multiple half-levels discard wavenumbers where all weights in a column are < 0)
    # require sum of weights = 1 by normalizing (Buehler), or determining Nth weight after first N - 1 (Moncet) (TODO?)
    
    # this doesn't yield a sum of 1? Not sure why it should? 
    no_negatives = False
    while(no_negatives == False):
        # can do without loop but will do with for now
        q_tilde = np.empty((num_hl, num_cols))
        w = np.empty((num_hl, len(S)))

        for i in range(num_hl):
            w[i] = np.linalg.lstsq(spectral_data.isel(wavenumber = S, half_level = i).data, reference_data.isel(half_level = i).data, rcond = None)[0]
        remove = w.max(axis = 0) < 0
        w[0][remove] = 0 #TODO : this is hardcoded for hl = 0
        S = np.delete(S, remove)
        if(not(remove.any() == True)): 
            no_negatives = True
                      
    # align new weights with original S
    w_final = np.zeros((num_hl, len(S_original)))
                       
    for h in range(num_hl):
        for n in range(len(S)):
            for i in range(len(S_original)):
                if (S[n] == S_original[i]):
                    w_final[h][i] = w[h][n]
    for j in range(num_hl):
        q_tilde[j] = np.matmul(w_final[j], (spectral_data.isel(wavenumber = S_original, half_level = j).data).T)
       
    #w_final = w_final/(np.sum(w_final))
    return w_final, q_v, q_tilde

def abs_rms(estimate, reference, num_hl):
    # absolute root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    return np.sqrt(((estimate - reference.data)**2).mean(axis = 1))

### TOA UP BROADBAND
lw = xr.open_dataset("/dx02/robertp/CKDMIP_LBL/evaluation1/lw_fluxes/ckdmip_evaluation1_lw_fluxes_present.h5",
                     engine = "netcdf4")

# Spectral flux data - fluxes per wavenumber at the TOA
spec_fluxes = xr.open_mfdataset(["/dx02/pc2943/spectral_fluxes_1-10.h5", "/dx02/pc2943/spectral_fluxes_11-20.h5",
                                "/dx02/pc2943/spectral_fluxes_21-30.h5", "/dx02/pc2943/spectral_fluxes_31-40.h5",
                                "/dx02/pc2943/spectral_fluxes_41-50.h5"], 
                                combine = 'nested', concat_dim = 'column',
                               engine = "netcdf4")
wavenumber_coords = spec_fluxes.wavenumber.data

# reference data at smaller interval should just be spectral sum over that interval, not whole atmosphere
TOA_up_spec = spec_fluxes.spectral_flux_up_lw.isel(half_level = 0).data
TOA_up_ref = TOA_up_spec.sum(axis = 1)
ref = np.array([TOA_up_ref.compute()])
ref = xr.DataArray(data = ref, dims = ["half_level", "column"], 
                  coords = dict(half_level=(["half_level"], np.array([0]))))

TOA_up_spec = TOA_up_spec.compute() 
TOA_up_spec = np.array(TOA_up_spec)
flux_subset = np.array([TOA_up_spec])
flux_subset = xr.DataArray(data = flux_subset, dims = ["half_level", "column", "wavenumber"],
                          coords = dict(half_level=(["half_level"], np.array([0])), 
                                        wavenumber=(["wavenumber"], wavenumber_coords)))

# Tidy the data for tree
input_data = pd.DataFrame(data=flux_subset.data[0], columns=flux_subset.wavenumber.data)
labels = ref.data[0]
features = np.array(input_data)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# trying grid search for best parameters
#params = {'max_depth': [6, 10],
#          'n_estimators': [10, 100],
#          'reg_alpha': [0, 5, 10, 20]
#}

#tree = xgboost.XGBRFRegressor(random_state = 42)
#clf = GridSearchCV(estimator = tree, param_grid = params, scoring = 'neg_mean_squared_error', verbose = 1)
#results = clf.fit(train_features, train_labels)
#joblib.dump(clf, 'clf.pkl')

# DIY grid search with different error estimate methods
tree_depth = np.array([6, 10])
num_trees = np.array([10, 100])
L1_reg = np.array([0, 5, 10, 20])

# save results
prediction_errors = np.zeros(16) # hardcoded for 16
feature_errors = np.zeros(16) # hardcoded for 16
idx = 0

for t in tree_depth:
    for n in num_trees:
        for l in L1_reg:
            
            # fit model with params
            rf = xgboost.XGBRFRegressor(n_estimators = n, random_state = 42, reg_alpha = l, max_depth = t, n_jobs = 28)
            rf.fit(train_features, train_labels)
            model_name = "tree" + str(t) + "num" + str(n) + "reg" + str(l) + ".pkl"
            joblib.dump(rf, model_name)
            prediction = rf.predict(test_features)
            
            # save error from predicting against test set 
            prediction_errors[idx] = np.sqrt(((prediction - test_labels)**2).mean())
            
            # find feature importances
            feature_list = list(np.arange(0, len(flux_subset.wavenumber.data)))
            importances = list(rf.feature_importances_)
            feature_importances = [(feature, importance) for feature, importance in zip(feature_list, importances)]
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            feature_importances = feature_importances[:100]
            
            # compute errors from reference calculations using most important features as representative set
            S = np.array([a_tuple[0] for a_tuple in feature_importances if a_tuple[1] > 0.0])
            w_final, q_v, q_tilde = compute_q_tilde(S, flux_subset, ref, 1, 50)
            
            # save error
            feature_errors[idx] = abs_rms(q_tilde, ref, 1)
            
            # increment index
            idx = idx + 1

# save errors to file
results = xr.Dataset(
    data_vars = dict(
        prediction_errors = prediction_errors,
        feature_errors = feature_errors, ), 
)

results.to_netcdf('results.h5')
            
               

