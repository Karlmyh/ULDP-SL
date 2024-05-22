import numpy as np
import os
import time
import scipy
import math
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ULDPFS import ULDPFS, ULDPFS_IA
from sklearn.linear_model import LassoCV

from distribution import TestDistribution



log_file_dir = "./logs/nm/"

def base_train_ULDPFS(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_y_list = [sample_generator.generate(m) for _ in range(n)]
    X_list = [X_y[0] for X_y in X_y_list]
    y_list = [X_y[1] for X_y in X_y_list]
    X_test, y_test = sample_generator.generate(2000)
    
    # ULDPFS
    method = "ULDPFS"
    regressor_params = {"num_bins": [2**1, 2**2, 2**3, 2**4, 2**5], "B":[1,2,3]}
    regressor_params_list = [dict(zip(regressor_params.keys(), value_product)) for value_product in product(*regressor_params.values())]
    param_dict = {"epsilon": [epsilon],
              "selector": ["postlasso", "screening"],
              "selector_params": [{"num_features" : 8 // 2, "screening_num_features" : 8**2 // 4}],
              "heavyhitter_params": [{"min_hitters": 8, "alpha" : 0.1}],
              "regressor_params": regressor_params_list}
    
    for param_values in product(*param_dict.values()):
        
        try:
            params = dict(zip(param_dict.keys(), param_values))

            # train
            time_start = time.time()
            model = ULDPFS(**params).fit(X_list, y_list)
            time_end = time.time()

            # evaluation
            ### variable selection
            target_var = sample_generator.nonzero_index
            pred_var = model.selected_indexes 
            # create indicators
            target_ind = np.zeros(d)
            target_ind[target_var] = 1
            pred_ind = np.zeros(d)
            pred_ind[pred_var] = 1
            
            # calculate accuracy, precision, recall using sklearn
            accuracy = accuracy_score(target_ind, pred_ind)
            precision = precision_score(target_ind, pred_ind)
            recall = recall_score(target_ind, pred_ind)
            
            ### efficiency
            time_used = time_end - time_start

            ### parameter estimation
            l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
            l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

            ### regression
            y_hat = model.predict(X_test) 
            mse = np.mean((y_hat - y_test)**2)

            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                
                logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                            dist_idx,
                                                                            method,
                                                                            iterate,
                                                                            epsilon,
                                                                            n,
                                                                            d, 
                                                                            m,
                                                                            accuracy,
                                                                            precision,
                                                                            recall,
                                                                            l2_error,
                                                                            l1_error,
                                                                            mse,
                                                                            "{}:{}:{}".format(params["selector"], params["regressor_params"]["num_bins"], params["regressor_params"]["B"]),
                                                                            time_used,
                                                                            )
            
                f.writelines(logs)
        except:
            pass
            
def base_train_ULDPFS_IA(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_y_list = [sample_generator.generate(m) for _ in range(n)]
    X_list = [X_y[0] for X_y in X_y_list]
    y_list = [X_y[1] for X_y in X_y_list]
    X_test, y_test = sample_generator.generate(2000)
    
    # ULDPFS_IA
    method = "ULDPFS_IA"
    regressor_params = {"num_bins": [2**3, 2**2, 2**4], "B":[3], "batch_size":[20, 40, 60], "lr_const":[0.1], "lr_power": [ 0.2]}
    regressor_params_list = [dict(zip(regressor_params.keys(), value_product)) for value_product in product(*regressor_params.values())]
    param_dict = {"epsilon": [epsilon],
              "selector": ["postlasso", "screening"],
              "selector_params": [{"num_features" : 8 // 2, "screening_num_features" : 8**2 // 4}],
              "heavyhitter_params": [{"min_hitters": 8, "alpha" : 0.1}],
              "regressor_params": regressor_params_list}
    
    for param_values in product(*param_dict.values()):
        
        try:
            params = dict(zip(param_dict.keys(), param_values))

            # train
            time_start = time.time()
            model = ULDPFS_IA(**params).fit(X_list, y_list)
            time_end = time.time()

            # evaluation
            ### variable selection
            target_var = sample_generator.nonzero_index
            pred_var = model.selected_indexes 
            # create indicators
            target_ind = np.zeros(d)
            target_ind[target_var] = 1
            pred_ind = np.zeros(d)
            pred_ind[pred_var] = 1
            
            # calculate accuracy, precision, recall using sklearn
            accuracy = accuracy_score(target_ind, pred_ind)
            precision = precision_score(target_ind, pred_ind)
            recall = recall_score(target_ind, pred_ind)
            
            ### efficiency
            time_used = time_end - time_start

            ### parameter estimation
            l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
            l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

            ### regression
            y_hat = model.predict(X_test) 
            mse = np.mean((y_hat - y_test)**2)

            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                
                logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                            dist_idx,
                                                                            method,
                                                                            iterate,
                                                                            epsilon,
                                                                            n,
                                                                            d, 
                                                                            m,
                                                                            accuracy,
                                                                            precision,
                                                                            recall,
                                                                            l2_error,
                                                                            l1_error,
                                                                            mse,
                                                                            "{}:{}:{}".format(params["selector"], params["regressor_params"]["batch_size"], params["regressor_params"]["num_bins"]),
                                                                            time_used,
                                                                            )
            
                f.writelines(logs)    
        except:
            pass
    

def base_train_NLDPSLR(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_train, y_train = sample_generator.generate(n * m)
    X_test, y_test = sample_generator.generate(2000)
    
    # NLDPSLR
    method = "NLDPSLR"
    param_dict =   {
        "r":[np.sqrt(d * np.log(n))],
        "tau1":[4],
        "tau2": [8],
        "lamda": [0.05],
        "epsilon": [epsilon],
            }
    
    for param_values in product(*param_dict.values()):
        
        params = dict(zip(param_dict.keys(), param_values))

        # train
        time_start = time.time()
        model = NLDPSLR(**params).fit(X_train, y_train)
        time_end = time.time()

        # evaluation
        ### variable selection
        target_var = sample_generator.nonzero_index
        pred_var = model.nonzero_indexes 
        # create indicators
        target_ind = np.zeros(d)
        target_ind[target_var] = 1
        pred_ind = np.zeros(d)
        pred_ind[pred_var] = 1
        
        # calculate accuracy, precision, recall using sklearn
        accuracy = accuracy_score(target_ind, pred_ind)
        precision = precision_score(target_ind, pred_ind)
        recall = recall_score(target_ind, pred_ind)
        
        ### efficiency
        time_used = time_end - time_start

        ### parameter estimation
        l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
        l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

        ### regression
        y_hat = model.predict(X_test) 
        mse = np.mean((y_hat - y_test)**2)

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                        dist_idx,
                                                                        method,
                                                                        iterate,
                                                                        epsilon,
                                                                        n,
                                                                        d, 
                                                                        m,
                                                                        accuracy,
                                                                        precision,
                                                                        recall,
                                                                        l2_error,
                                                                        l1_error,
                                                                        mse,
                                                                        "{}:{}:{}:{}".format(params["r"], params["tau1"], params["tau2"], params["lamda"]),
                                                                        time_used,
                                                                        )
            f.writelines(logs)
            

def base_train_LDPIHT(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_train, y_train = sample_generator.generate(n * m)
    X_test, y_test = sample_generator.generate(2000)
    
    # LDPIHT
    method = "LDPIHT"
    param_dict =   {
        "eta":[0.01, 0.1, 1],
        "s":[5, 10, 20, 50],
        "T": [10, 20, 50],
        "epsilon": [epsilon],
            }
    
    for param_values in product(*param_dict.values()):
        
        params = dict(zip(param_dict.keys(), param_values))

        # train
        time_start = time.time()
        model = LDPIHT(**params).fit(X_train, y_train)
        time_end = time.time()

        # evaluation
        ### variable selection
        target_var = sample_generator.nonzero_index
        pred_var = model.nonzero_indexes 
        # create indicators
        target_ind = np.zeros(d)
        target_ind[target_var] = 1
        pred_ind = np.zeros(d)
        pred_ind[pred_var] = 1
        
        # calculate accuracy, precision, recall using sklearn
        accuracy = accuracy_score(target_ind, pred_ind)
        precision = precision_score(target_ind, pred_ind)
        recall = recall_score(target_ind, pred_ind)
        
        ### efficiency
        time_used = time_end - time_start

        ### parameter estimation
        l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
        l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

        ### regression
        y_hat = model.predict(X_test) 
        mse = np.mean((y_hat - y_test)**2)

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                        dist_idx,
                                                                        method,
                                                                        iterate,
                                                                        epsilon,
                                                                        n,
                                                                        d, 
                                                                        m,
                                                                        accuracy,
                                                                        precision,
                                                                        recall,
                                                                        l2_error,
                                                                        l1_error,
                                                                        mse,
                                                                        "{}:{}:{}".format(params["T"], params["s"], params["eta"]),
                                                                        time_used,
                                                                        )
            f.writelines(logs)


def base_train_LASSO(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_train, y_train = sample_generator.generate(n * m)
    X_test, y_test = sample_generator.generate(2000)
    
    # LASSO
    method = "LASSO"

    # train
    time_start = time.time()
    model = LassoCV(fit_intercept = False).fit(X_train, y_train)
    time_end = time.time()

    # evaluation
    ### variable selection
    target_var = sample_generator.nonzero_index
    pred_var = np.nonzero(model.coef_)[0]
    # create indicators
    target_ind = np.zeros(d)
    target_ind[target_var] = 1
    pred_ind = np.zeros(d)
    pred_ind[pred_var] = 1
    
    # calculate accuracy, precision, recall using sklearn
    accuracy = accuracy_score(target_ind, pred_ind)
    precision = precision_score(target_ind, pred_ind)
    recall = recall_score(target_ind, pred_ind)
    
    ### efficiency
    time_used = time_end - time_start

    ### parameter estimation
    l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
    l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

    ### regression
    y_hat = model.predict(X_test) 
    mse = np.mean((y_hat - y_test)**2)

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                    dist_idx,
                                                                    method,
                                                                    iterate,
                                                                    epsilon,
                                                                    n,
                                                                    d, 
                                                                    m,
                                                                    accuracy,
                                                                    precision,
                                                                    recall,
                                                                    l2_error,
                                                                    l1_error,
                                                                    mse,
                                                                    "",
                                                                    time_used,
                                                                    )
        f.writelines(logs)
            
            
            
            
def base_train_LOCALLASSO(iterate, dist_idx, epsilon, n, d, m):
    
    np.random.seed(iterate)
    sample_generator = TestDistribution(dist_idx, d).returnDistribution()
    X_train, y_train = sample_generator.generate( m)
    X_test, y_test = sample_generator.generate(2000)
    
    # LASSO
    method = "LOCALLASSO"

    # train
    time_start = time.time()
    model = LassoCV(fit_intercept = False).fit(X_train, y_train)
    time_end = time.time()

    # evaluation
    ### variable selection
    target_var = sample_generator.nonzero_index
    pred_var = np.nonzero(model.coef_)[0]
    # create indicators
    target_ind = np.zeros(d)
    target_ind[target_var] = 1
    pred_ind = np.zeros(d)
    pred_ind[pred_var] = 1
    
    # calculate accuracy, precision, recall using sklearn
    accuracy = accuracy_score(target_ind, pred_ind)
    precision = precision_score(target_ind, pred_ind)
    recall = recall_score(target_ind, pred_ind)
    
    ### efficiency
    time_used = time_end - time_start

    ### parameter estimation
    l2_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 2)
    l1_error = np.linalg.norm(model.coef_ - sample_generator.beta, ord = 1)

    ### regression
    y_hat = model.predict(X_test) 
    mse = np.mean((y_hat - y_test)**2)

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{},{:4f}\n".format(
                                                                    dist_idx,
                                                                    method,
                                                                    iterate,
                                                                    epsilon,
                                                                    n,
                                                                    d, 
                                                                    m,
                                                                    accuracy,
                                                                    precision,
                                                                    recall,
                                                                    l2_error,
                                                                    l1_error,
                                                                    mse,
                                                                    0,
                                                                    time_used,
                                                                    )
        f.writelines(logs)
        
        
def run_nm_simulation():
    num_repetitions = 30  
    num_jobs = 30
    
    nm = 600 * 200
    n_m_ratio_vec = [0.09375, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24]
    
    for dist_idx in [1, 3]:
        for d in [   256]:
            for n_m_ratio in n_m_ratio_vec:
                m =  int(np.sqrt(nm / n_m_ratio))
                n = nm // m
                for epsilon in [ 4]: 
                    
                    break_point = (0,0,0,0)
                    if epsilon < break_point[0] or n < break_point[1] or d < break_point[2] or m < break_point[3]:
                        continue
                    print(epsilon, n, d, m)
                    # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_ULDPFS)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LDPIHT)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_NLDPSLR)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_ULDPFS_IA)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LASSO)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LOCALLASSO)(i, dist_idx, epsilon, n, d, m) for i in range(num_repetitions))
                    
                    
                    
              