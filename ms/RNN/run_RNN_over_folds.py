import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.model_selection import train_test_split
import os
from ms.models.RNN import train_rnn_ms, files_linking
from ms import DATA_DIR

import argparse
import wandb
folds = range(0,5)

gpu_num = 1
device = torch.device(f"cuda")
sim_type = "binned"
tensor_type = "mat"
#torch.cuda.set_device(gpu_num)
suffix = "_original"

hyper_dict = {"num_layers":[1,2],
            "classification_hidden":[10,50],
            "weight_decay":[0.0001,0.001],#,0.01,0.1,1],
            "dropout_rate":[0,0.1],
            "cov_hidden" : [25,50],
            "hidden_size" : [50,100],
            "learning_rate": [0.001, 0.005]}

keys = hyper_dict.keys()
values = (hyper_dict[key] for key in keys)
combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

for fold in folds:

    print(f"Computing fold {fold}")
    train_idx = np.load(DATA_DIR + f"/folds/train_idx_{fold}.npy",allow_pickle = True)
    val_idx = np.load(DATA_DIR + f"/folds/val_idx_{fold}.npy", allow_pickle = True)
    test_idx = np.load(DATA_DIR + f"/folds/test_idx_{fold}.npy", allow_pickle = True)

    for c in combinations:
        dropout_rate = c["dropout_rate"]
        classification_hidden = c["classification_hidden"]
        weight_decay = c["weight_decay"]
        num_layers = c["num_layers"]
        cov_hidden = c["cov_hidden"]
        hidden_size = c["hidden_size"]
        learning_rate = c["learning_rate"]
        

        simulation_name = f"21_original_residual_Xval_RNN_MS_confirmed_{tensor_type}_{sim_type}_D{dropout_rate}_CH{classification_hidden}_L2{weight_decay}_NL{num_layers}_CH{cov_hidden}_HS{hidden_size}_fold{fold}" 


        #Model parameters.
        params_dict=dict()
        params_dict["hidden_size"] = hidden_size
        params_dict["delta_t"]= 0.1
        params_dict["T"]= 36
        params_dict["residual_classifier"] = True

        params_dict["classification_hidden"] = classification_hidden
        params_dict["cov_hidden"] = cov_hidden
        params_dict["weight_decay"] = weight_decay
        params_dict["dropout"] = dropout_rate
        params_dict["lr"]= learning_rate
        params_dict["full_gru_ode"] = True
        params_dict["no_cov"] = False
        params_dict["simulation_name"] = simulation_name
        params_dict["num_layers"] = num_layers

        csv_files = files_linking(sim_type, tensor_type, params_dict, suffix = suffix)

        info, val_metric_prev,val_metric_pr, test_auc, test_auc_pr = train_rnn_ms(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        csv_files = csv_files,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=100)

        params_dict["Best_val_AUC"] = val_metric_prev
        params_dict["Best_test_AUC"] = test_auc
        params_dict["Best_val_AUCPR"] = val_metric_pr
        params_dict["Best_test_AUCPR"] = test_auc_pr

        #params_dict["Last_val_AUC"] = info["AUC_validation"]
        output_file = f"./trained_models/21_Xval_RNN_fold{fold}.csv"
        current_res = pd.DataFrame.from_records(params_dict,index = [0])
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df = df.append(current_res,sort =False)
            df.to_csv(output_file,index = False)
        else:
            current_res.to_csv(output_file, index = False)

