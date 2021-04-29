import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.model_selection import train_test_split
import os
import argparse
from ms.models.run_GRUODE import train_gruode_ms, files_linking
folds = range(5)
folds = [0,1,2,3,4]
from ms import DATA_DIR

#gpu_num = 1
device = torch.device(f"cuda")
sim_type = "binned"
tensor_type = "mat"
#torch.cuda.set_device(gpu_num)

hyper_dict = {"prep_hidden":[4,10],
            "classification_hidden":[20,50],
            "weight_decay":[0.000001,0.00001,0.0001],
            "dropout_rate":[0,0.1],
            "learning_rate": [0.01,0.001]}

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
        prep_hidden = c["prep_hidden"]
        learning_rate = c["learning_rate"]

        simulation_name = f"21_Xval_GRUODE_confirmedNEW3_MS_{tensor_type}_{sim_type}_D{dropout_rate}_CH{classification_hidden}_L2{weight_decay}_PH{prep_hidden}_lr{learning_rate}_fold{fold}" 


        #Model parameters.
        params_dict=dict()
        params_dict["hidden_size"] = 50
        params_dict["p_hidden"] = 10
        params_dict["prep_hidden"] = prep_hidden
        params_dict["logvar"] = True
        params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
        params_dict["delta_t"]= 0.1
        params_dict["T"]=36
        params_dict["lambda"] = 100000 #Weighting between classification and MSE loss.

        params_dict["classification_hidden"] = classification_hidden
        params_dict["cov_hidden"] = 25
        params_dict["weight_decay"] = weight_decay
        params_dict["dropout_rate"] = dropout_rate
        params_dict["lr"]= learning_rate
        params_dict["full_gru_ode"] = True
        params_dict["no_cov"] = False
        params_dict["simulation_name"] = simulation_name

        csv_files = files_linking(sim_type, tensor_type, params_dict)

        info, val_metric_prev, test_loglik, test_auc, test_mse = train_gruode_ms(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        csv_files = csv_files,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=100)

        params_dict["Best_val_AUC"] = val_metric_prev
        params_dict["Best_test_AUC"] = test_auc
        params_dict["Last_val_AUC"] = info["AUC_validation"]
        output_file = f"./trained_models/Xval_GRUODE_fold{fold}.csv"
        current_res = pd.DataFrame.from_records(params_dict,index = [0])
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            df = df.append(current_res,sort =False)
            df.to_csv(output_file,index = False)
        else:
            current_res.to_csv(output_file, index = False)

            
            




