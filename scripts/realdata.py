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


data_file_dir = "./data/cleaned/"
log_file_dir = "./logs/realdata/"


# collect real data samples as the datasets of each user
def whole_dataset_collector(data_name, test_portion, iterate):
    file_name = "{}.csv".format(data_name)
    path = os.path.join(data_file_dir, file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    group_idx = data[:, -1]
    X = data[:,1:-1]
    y = data[:,0]
    
    # sample the data of users to be in the test set
    np.random.seed(iterate)
    test_idx = np.random.choice(len(y), int(len(y) * test_portion), replace = False)
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, y_train, X_test, y_test
        
 # collect real data samples as a whole dataset       
def grouped_dataset_collector(data_name, test_portion, iterate):
    file_name = "{}.csv".format(data_name)
    path = os.path.join(data_file_dir, file_name)
    data = pd.read_csv(path, header=None)
    data = np.array(data, dtype = "float")
    group_idx = data[:, -1]
    X = data[:,1:-1]
    y = data[:,0]
    
    
    
    unique_group_idx = np.unique(group_idx)
    # sample the data of users to be in the test set
    np.random.seed(iterate)
    test_group_idx = np.random.choice(unique_group_idx, int(len(unique_group_idx) * test_portion), replace = False)
    train_group_idx = np.setdiff1d(unique_group_idx, test_group_idx)
    
    X_list = []
    y_list = []
    for group in train_group_idx:
        idx = np.where(group_idx == group)[0]
        X_group = X[idx]
        y_group = y[idx]
        X_list.append(X_group)
        y_list.append(y_group)
        
    # concencate all the test sets
    idx = np.where(np.isin(group_idx, test_group_idx))[0]
    X_test = X[idx]
    y_test = y[idx]
    

    
    return X_list, y_list, X_test, y_test



def base_train_ULDPFS(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_list, y_list, X_test, y_test = grouped_dataset_collector(data_name, 0.2, iterate)
    # ULDPFS
    method = "ULDPFS"
    regressor_params = {"num_bins": [2**1, 2**2, 2**3, 2**4, 2**5], "B":[1,2,3]}
    regressor_params_list = [dict(zip(regressor_params.keys(), value_product)) for value_product in product(*regressor_params.values())]
    selector_params = {"num_features" : [2, 4, 8 , 16], "screening_num_features" : [16**2 // 4]}
    selector_params_list = [dict(zip(selector_params.keys(), value_product)) for value_product in product(*selector_params.values())]
    heavyhitter_params = {"min_hitters": [2, 4, 8 , 16], "alpha" : [0.1]}
    heavyhitter_params_list = [dict(zip(heavyhitter_params.keys(), value_product)) for value_product in product(*heavyhitter_params.values())]
    param_dict = {"epsilon": [epsilon],
              "selector": [   "postlasso"],
              "selector_params": selector_params_list, 
              "heavyhitter_params": heavyhitter_params_list,
              "regressor_params": regressor_params_list}
    
    for param_values in product(*param_dict.values()):
        
        try:
            params = dict(zip(param_dict.keys(), param_values))

            # train
            time_start = time.time()
            model = ULDPFS(**params).fit(X_list, y_list)
            time_end = time.time()

            # evaluation
            ### efficiency
            time_used = time_end - time_start
            ### regression
            y_hat = model.predict(X_test) 
            mse = np.mean((y_hat - y_test)**2)

            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                
                logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                        data_name,
                                                        method,
                                                        iterate,
                                                        epsilon,
                                                        mse,
                                                        "{}:{}:{}:{}:{}".format(params["selector"], params["regressor_params"]["num_bins"], params["regressor_params"]["B"], params["selector_params"]["num_features"], params["heavyhitter_params"]["min_hitters"]),
                                                        time_used,
                                                        )
            
                f.writelines(logs)
        except:
            pass
            
def base_train_ULDPFS_IA(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_list, y_list, X_test, y_test = grouped_dataset_collector(data_name, 0.2, iterate)
    # ULDPFS_IA
    method = "ULDPFS_IA"
    regressor_params = {"num_bins": [2**3, 2**2], "B":[3], "batch_size":[10, 20, 40, 60], "lr_const":[0.1], "lr_power": [ 0.2]}
    regressor_params_list = [dict(zip(regressor_params.keys(), value_product)) for value_product in product(*regressor_params.values())]
    selector_params = {"num_features" : [2, 4, 8 , 16], "screening_num_features" : [16**2 // 4]}
    selector_params_list = [dict(zip(selector_params.keys(), value_product)) for value_product in product(*selector_params.values())]
    heavyhitter_params = {"min_hitters": [2, 4, 8 , 16], "alpha" : [0.1]}
    heavyhitter_params_list = [dict(zip(heavyhitter_params.keys(), value_product)) for value_product in product(*heavyhitter_params.values())]
    param_dict = {"epsilon": [epsilon],
              "selector": [ "screening", "postlasso"],
              "selector_params": selector_params_list,
              "heavyhitter_params": heavyhitter_params_list,
              "regressor_params": regressor_params_list}
    
    for param_values in product(*param_dict.values()):
        
        try:
            params = dict(zip(param_dict.keys(), param_values))

            # train
            time_start = time.time()
            model = ULDPFS_IA(**params).fit(X_list, y_list)
            time_end = time.time()
            ### efficiency
            time_used = time_end - time_start
            ### regression
            y_hat = model.predict(X_test) 
            mse = np.mean((y_hat - y_test)**2)

            log_file_name = "{}.csv".format(method)
            log_file_path = os.path.join(log_file_dir, log_file_name)
            with open(log_file_path, "a") as f:
                
                logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                        data_name,
                                                        method,
                                                        iterate,
                                                        epsilon,
                                                        mse,
                                                        "{}:{}:{}:{}:{}".format(params["selector"], params["regressor_params"]["num_bins"], params["regressor_params"]["batch_size"], params["selector_params"]["num_features"], params["heavyhitter_params"]["min_hitters"]),
                                                        time_used,
                                                        )
            
                f.writelines(logs)  
        except:
            pass  
    
    

def base_train_NLDPSLR(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_train, y_train, X_test, y_test = whole_dataset_collector(data_name, 0.2, iterate)
    n, d = X_train.shape
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
        ### efficiency
        time_used = time_end - time_start
        ### regression
        y_hat = model.predict(X_test) 
        mse = np.mean((y_hat - y_test)**2)

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                    data_name,
                                                    method,
                                                    iterate,
                                                    epsilon,
                                                    mse,
                                                    "{}:{}:{}:{}".format(params["r"], params["tau1"], params["tau2"], params["lamda"]),
                                                    time_used,
                                                    )
            f.writelines(logs)
            

def base_train_LDPIHT(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_train, y_train, X_test, y_test = whole_dataset_collector(data_name, 0.2, iterate)
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
        ### efficiency
        time_used = time_end - time_start

        ### regression
        y_hat = model.predict(X_test) 
        mse = np.mean((y_hat - y_test)**2)

        log_file_name = "{}.csv".format(method)
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                    data_name,
                                                    method,
                                                    iterate,
                                                    epsilon,
                                                    mse,
                                                    "{}:{}:{}".format(params["T"], params["s"], params["eta"]),
                                                    time_used,
                                                    )
            f.writelines(logs)


def base_train_LASSO(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_train, y_train, X_test, y_test = whole_dataset_collector(data_name, 0.2, iterate)
    # LASSO
    method = "LASSO"
    # train
    time_start = time.time()
    model = LassoCV(fit_intercept = False).fit(X_train, y_train)
    time_end = time.time()

    # evaluation
    ### efficiency
    time_used = time_end - time_start
    ### regression
    y_hat = model.predict(X_test) 
    mse = np.mean((y_hat - y_test)**2)

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                data_name,
                                                method,
                                                iterate,
                                                epsilon,
                                                mse,
                                                "0",
                                                time_used,
                                                )
        f.writelines(logs)
        
def base_train_LOCALLASSO(iterate, data_name, epsilon):
    
    np.random.seed(iterate)
    X_list, y_list, X_test, y_test = grouped_dataset_collector(data_name, 0.2, iterate)
    # LASSO
    method = "LOCALLASSO"
    # train
    time_start = time.time()
    model = LassoCV(fit_intercept = False).fit(X_list[0], y_list[0])
    time_end = time.time()

    # evaluation
    ### efficiency
    time_used = time_end - time_start
    ### regression
    y_hat = model.predict(X_test) 
    mse = np.mean((y_hat - y_test)**2)

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{:4f},{},{}\n".format(
                                                data_name,
                                                method,
                                                iterate,
                                                epsilon,
                                                mse,
                                                "0",
                                                time_used,
                                                )
        f.writelines(logs)
            
            
def run_base_realdata():
    num_repetitions = 30  
    num_jobs = 30
    
    data_file_name_seq = [
    "airlines",
    "loandefault",
    "mip",
    "wine",
    "yolanda",
    "taxi"
    ]
    

    
    
    for idx_data_name, data_name in enumerate(data_file_name_seq):
        for epsilon in [  1, 4]: 
            break_point = (0, 0)
            if idx_data_name < break_point[0] or epsilon < break_point[1]:
                continue
            print(data_name, epsilon)
            Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_ULDPFS)(i, data_name, epsilon) for i in range(num_repetitions))
            # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LDPIHT)(i, data_name, epsilon) for i in range(num_repetitions))
            # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_NLDPSLR)(i, data_name, epsilon) for i in range(num_repetitions))
            # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_ULDPFS_IA)(i, data_name, epsilon) for i in range(num_repetitions))
            # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LASSO)(i, data_name, epsilon) for i in range(num_repetitions))
            # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_train_LOCALLASSO)(i, data_name, epsilon) for i in range(num_repetitions))
   
                        