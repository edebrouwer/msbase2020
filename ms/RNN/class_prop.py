"""

Classification propagation.

"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gru_ode
import gru_ode.data_utils as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
from gru_ode import Logger
import os
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def class_prob(simulation_name, device, csv_files, test_idx, epoch_max=40, batch_size = 1, logits = False, calibrator = None, mscourse = None):

    csv_file_path = csv_files["path"]
    csv_file_tags = csv_files["tags"]
    csv_file_cov  = csv_files["cov"]

    N = pd.read_csv(csv_file_tags)["ID"].nunique()

    validation = True
    
    if "cont" in csv_file_path:
        val_options = {"T_val": 1095, "max_val_samples": 1, "T_closest": 1825, "T_val_from" : 1460}
    else:
        val_options = {"T_val": 36, "max_val_samples": 1, "T_closest":60, "T_val_from" : 48}

    if mscourse is not None:
        df_cov = pd.read_csv(csv_file_cov)
        test_idx = np.array(df_cov.loc[(df_cov.ID.isin(test_idx)) & ((df_cov[mscourse]>0).any(1)),"ID"].unique().tolist())
    
    data_test = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags, cov_file= csv_file_cov, idx=test_idx,validation = validation, val_options = val_options)

    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=False, batch_size=batch_size)


    params_dict = np.load(f"./trained_models/{simulation_name}_params.npy", allow_pickle = True).item()

    nnfwobj = gru_ode.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            classification_hidden=params_dict["classification_hidden"],
                                            cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                            dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"])
    nnfwobj.to(device)

    nnfwobj.load_state_dict(torch.load(f"./trained_models/{simulation_name}_MAX.pt")) 

    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    val_metric_prev = -1000
    nnfwobj.eval()
    class_preds = []
    labels_list = []
    for i, b in enumerate(tqdm.tqdm(dl_test)):

        prob_path = []
        times    = b["times"]
        time_ptr = b["time_ptr"]
        X        = b["X"].to(device)
        M        = b["M"].to(device)
        obs_idx  = b["obs_idx"]
        cov      = b["cov"].to(device)
        labels = b["y"].to(device)
        batch_size = labels.size(0)

        if labels.shape[0]>1:
            _,_,class_pred,_ = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov)

            if logits:
                return labels.detach().cpu().numpy(),class_pred.detach().cpu().numpy()
            else:
                return labels.detach().cpu().numpy(), torch.sigmoid(class_pred).detach().cpu().numpy()


        for samp in range(0,len(times)+1):
            times_samp = times[:samp]
            time_ptr_samp = time_ptr[:samp]
            X_samp = X[:samp]
            M_samp = M[:samp]
            obs_idx_samp = obs_idx[:samp]

            hT, loss, class_pred, _  = nnfwobj(times, time_ptr, X_samp, M_samp, obs_idx_samp, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov)

            prob_path += [clf.predict_proba((class_pred).detach().cpu())[:,1]] 
        
        class_preds += [class_pred.detach().cpu().numpy().item()]
        labels_list += [labels.detach().cpu().numpy().item()]

        
        plt.figure()
        times /= 12 
        times -= 3
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time before visit [Years]')
        ax1.set_ylabel('EDSS', color=color)
        edss_x = np.round(2*(X.detach().cpu().numpy()*1.6764+2.4818))/2
        ax1.scatter(times, edss_x, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(edss_x.min()-1,edss_x.max()+1)
        min_tick = np.max((0,edss_x.min()-0.5))
        max_tick = np.min((10,edss_x.max()+1))
        ax1.set_yticks(np.arange(min_tick,max_tick,step = 0.5))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Probability', color=color)  # we already handled the x-label with ax1
        ax2.step(np.concatenate((np.array([-3]),times)), prob_path, where = "post", color = color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim((0,1))
        fig.tight_layout()
        
        #plt.scatter(times,X.detach().cpu().numpy())
        #plt.step(np.concatenate((np.array([0]),times)), prob_path, where = "post")
        plt.title(f"Progression of the worsening prediction over time. Label : {labels.detach().cpu().numpy()[0][0]}")
        fig.savefig(f"./figs/prob_prop_{i}.pdf")
        plt.close(fig)
        plt.close("all")

        

        #if i >100:
        #    break
    print(roc_auc_score(np.array(labels_list),np.array(class_preds)))

    return class_preds, labels_list
    #return(info, val_metric_prev, test_loglik, test_auc, test_mse)

def files_linking(sim_type, tensor_type, params_dict, suffix = ""):
    if sim_type == "continuous": 
        csv_file_path = DATA_DIR +  f"/RNN/{tensor_type}_data_cont{suffix}.csv"
        csv_file_tags = DATA_DIR + f"/RNN/label_data_cont{suffix}.csv"
        if params_dict["no_cov"]:
            csv_file_cov = None
        else:
            csv_file_cov  = DATA_DIR + f"/RNN/cov_data_cont{suffix}.csv"
    elif sim_type == "binned":
        csv_file_path = DATA_DIR + f"/RNN/{tensor_type}_data{suffix}.csv"
        csv_file_tags = DATA_DIR + f"/RNN/label_data{suffix}.csv"
        if params_dict["no_cov"]:
            csv_file_cov = None
        else:
            csv_file_cov  = DATA_DIR + f"/RNN/cov_data{suffix}.csv"
    else:
        raise "invalid simtype"
    
    csv_files = dict()
    csv_files["path"] = csv_file_path
    csv_files["tags"] = csv_file_tags
    csv_files["cov"]  = csv_file_cov
    return csv_files

if __name__ =="__main__":

    gpu_num = 1
    sim_type = "binned"
    tensor_type = "mat"

    #simulation_name=f"MS_{sim_type}_{tensor_type}_no_cov"
    simulation_name = "Xval_GRUODE_confirmedNEW3_MS_mat_binned_D0_CH50_L20.0001_PH10_fold1"
    
    device = torch.device(f"cuda:{gpu_num}")
    device = torch.device("cpu")
    torch.cuda.set_device(gpu_num)

    params_dict = np.load(f"./trained_models/{simulation_name}_params.npy", allow_pickle = True).item()
    
    train_idx = np.load(DATA_DIR + "/folds/train_idx_1.npy", allow_pickle = True)
    val_idx = np.load(DATA_DIR + "/folds/val_idx_1.npy", allow_pickle = True)
    test_idx = np.load(DATA_DIR + "/folds/test_idx_1.npy", allow_pickle  = True)

    #Model parameters.
   
    csv_files = files_linking(sim_type, tensor_type, params_dict)

    #csv_files["cov"] = "./sub_cov_data.csv"

    batch_size = len(val_idx)

    labels_val, logits_val = class_prob(simulation_name = simulation_name,
                        device = device,
                        csv_files = csv_files,
                        test_idx = val_idx,
                        epoch_max=100, batch_size = batch_size, logits = True)

    clf = LogisticRegression(random_state=0).fit(logits_val, labels_val[:,0])

    info, val_metric_prev, test_loglik, test_auc, test_mse = class_prob(simulation_name = simulation_name,
                        device = device,
                        csv_files = csv_files,
                        test_idx = test_idx,
                        epoch_max=100, batch_size = 1, calibrator = clf)
            

