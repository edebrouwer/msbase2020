import argparse
import ms.RNN.data_utils as data_utils
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
#from gru_ode import Logger
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import time

from ms import DATA_DIR


def train_rnn_ms(simulation_name, params_dict,device, csv_files, train_idx, val_idx, test_idx, epoch_max=40):

    csv_file_path = csv_files["path"]
    csv_file_tags = csv_files["tags"]
    csv_file_cov  = csv_files["cov"]


    N = pd.read_csv(csv_file_tags)["ID"].nunique()

    #logger = Logger(f'../../Logs/MS/{simulation_name}')

    validation = True
    
    if "cont" in csv_file_path:
        val_options = {"T_val": 1095, "max_val_samples": 1, "T_closest": 1825, "T_val_from" : 1460}
    else:
        val_options = {"T_val": 36, "max_val_samples": 1, "T_closest":60, "T_val_from" : 48}
    
    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags, cov_file= csv_file_cov, idx=train_idx,validation = validation, val_options = val_options, root_dir = "")
    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=val_idx, validation = validation,
                                        val_options = val_options,root_dir = "")
    data_test   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options, root_dir = "")
    
    
    dl   = DataLoader(dataset=data_train, collate_fn=data_utils.discrete_collate_fn, shuffle=True, batch_size=1000,num_workers=4)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.discrete_collate_fn, shuffle=True, batch_size=len(val_idx))
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.discrete_collate_fn, shuffle=True, batch_size=len(test_idx))

    params_dict["input_size"]=data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim

    model_name = simulation_name

    np.save(f"./trained_models/{simulation_name}_params.npy",params_dict)

    pos_weight = torch.tensor(((data_train.label_df.shape[0]/data_train.label_df.sum())-1).values,device = device, dtype = torch.float)
    print(f"Pos weight : {pos_weight}")
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum',pos_weight = pos_weight)


    ## model definition
    model = RNN_classification_model(params_dict)
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=params_dict["lr"], weight_decay= params_dict["weight_decay"])


    #Training
    best_val_auc = 0
    best_val_auc_pr = 0
    last_best_epoch = 0
    for epoch in range(epoch_max):
        if epoch-last_best_epoch > 20:
            break
        print(f"Epoch----{epoch} out of {epoch_max}")
        model.train()
        total_train_loss = 0
        complication_pred = np.zeros(len(data_train))
        complication_label = np.zeros(len(data_train))
        n_pat_processed = 0
        for i, b in tqdm.tqdm(enumerate(dl)):


            optimizer.zero_grad()
            X        = b["X"].to(device)
            cov      = b["cov"].to(device)
            y = b["y"].to(device)


            class_pred = model(X,cov)

            loss = class_criterion(class_pred,y)
            loss.backward()
            optimizer.step()

            complication_pred[n_pat_processed:(n_pat_processed+len(y))] = class_pred[:,0].cpu().detach().numpy()
            complication_label[n_pat_processed:(n_pat_processed+len(y))] = y[:,0].cpu().detach().numpy()
            n_pat_processed += len(y)
            total_train_loss += loss.cpu().detach().numpy()

        auc_train = roc_auc_score(complication_label,complication_pred)
        info = {"auc_train": auc_train, "Loss_train": total_train_loss/len(data_train)}
        print(f"AUC train : {auc_train}")
        print(f"Training_loss : {total_train_loss}")

        auc_val,_, _, _, auc_pr_val = evaluate_model(model, data_val, dl_val, device)
        info['auc_val'] = auc_val
        print(f" AUC for Event of Interest : {auc_val} - PR {auc_pr_val}")

        #for tag, value in info.items():
        #    logger.scalar_summary(tag,value,epoch)

        if auc_val > best_val_auc:
           print("New best AUC-PR on validation set : saving the model...")
           torch.save(model.state_dict(),f"./trained_models/{model_name}_best_val.pt")
           best_val_auc = auc_val
           best_auc_train = auc_train
           best_val_auc_pr = auc_pr_val
           auc_test, _, _, _, auc_pr_test = evaluate_model(model, data_test,dl_test, device)
           print(f"Test AUC : {auc_test} - PR {auc_pr_test}")
           last_best_epoch = epoch
        if (epoch % 20) == 0:
           torch.save(model.state_dict(),f"./trained_models/{model_name}.pt")

    

    df_file_name = "./trained_models/RNN_results.csv"
    df_res = pd.DataFrame({"Name" : [model_name], "AUC_test" : [auc_test], "AUC_val" : [best_val_auc], "AUC_PR_val": [auc_pr_val], "AUC_pr_test": [auc_pr_test], "AUC_train" : [best_auc_train]})
    if os.path.isfile(df_file_name):
        df = pd.read_csv(df_file_name)
        df = df.append(df_res)
        df.to_csv(df_file_name,index=False)
    else:
        df_res.to_csv(df_file_name,index=False)


    return info, best_val_auc, best_val_auc_pr, auc_test, auc_pr_test



def evaluate_model(model, data_val, dl_val,device):
    with torch.no_grad():
        loss_val = 0
        num_obs  = 0
        model.eval()
        complication_pred = np.zeros(len(data_val))
        complication_label = np.zeros(len(data_val))
        n_pat_processed = 0
        indices = np.zeros(len(data_val))
        
        for i, b in enumerate(dl_val):
            X        = b["X"].to(device)
            cov      = b["cov"].to(device)
            y = b["y"]

            class_pred = model(X,cov)
            complication_pred[n_pat_processed:(n_pat_processed+len(y))] = class_pred[:,0].cpu().detach().numpy()
            complication_label[n_pat_processed:(n_pat_processed+len(y))] = y[:,0].cpu().detach().numpy()
            indices[n_pat_processed:(n_pat_processed+len(y))] = np.array(b["indices"])
            n_pat_processed += len(y)

        auc_val = roc_auc_score(complication_label, complication_pred)
        precision, recall, _ = precision_recall_curve(complication_label, complication_pred)
        auc_pr_val = auc(recall,precision)

        return auc_val, complication_pred, complication_label, indices, auc_pr_val

class View(torch.nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)


class RNN_classification_model(torch.nn.Module):
    def __init__(self, params_dict):
        super().__init__()
        self.RNN_mod = torch.nn.GRU(num_layers = params_dict["num_layers"], input_size = 1+2*params_dict["input_size"],hidden_size = params_dict["hidden_size"], dropout = params_dict["dropout"])
 
        self.residual_classifier = params_dict.get("residual_classifier",False) 
       
        if self.residual_classifier:
            self.classification_model =  torch.nn.Sequential(torch.nn.Linear(params_dict["hidden_size"] + params_dict["cov_size"] , int(params_dict["classification_hidden"])), torch.nn.ReLU(),torch.nn.Dropout(params_dict["dropout"]) ,torch.nn.Linear(int(params_dict["classification_hidden"]),1))
        else:
            self.classification_model = torch.nn.Sequential(torch.nn.Linear(params_dict["hidden_size"],int(params_dict["classification_hidden"])), torch.nn.ReLU(),torch.nn.Dropout(params_dict["dropout"]) ,torch.nn.Linear(int(params_dict["classification_hidden"]),1))


        self.cov_map = torch.nn.Sequential(torch.nn.Linear(params_dict["cov_size"], params_dict["cov_hidden"]), torch.nn.Tanh(), torch.nn.Linear(params_dict["cov_hidden"],params_dict["num_layers"]*params_dict["hidden_size"]), torch.nn.Tanh(), View(params_dict["num_layers"],-1,params_dict["hidden_size"]))


    def forward(self,x,covs):
        h0 = self.cov_map(covs)
        _, hT = self.RNN_mod(x,h0)
        
        if self.residual_classifier:
            class_pred = self.classification_model(torch.cat((hT[-1],covs),1))
        else:
            class_pred = self.classification_model(hT[-1,:,:])
        return class_pred

def files_linking(sim_type, tensor_type, params_dict, suffix = ""):
    if sim_type == "continuous": 
        csv_file_path = DATA_DIR + "/RNN"  + f"/{tensor_type}_data_cont{suffix}.csv"
        csv_file_tags = DATA_DIR + "/RNN"  + f"/label_data{suffix}.csv"
        if params_dict["no_cov"]:
            csv_file_cov = None
        else:
            csv_file_cov  = DATA_DIR + "/RNN"  +  "/cov_data{suffix}.csv"
    elif sim_type == "binned":
        csv_file_path = DATA_DIR + "/RNN"  +  f"/{tensor_type}_data{suffix}.csv"
        csv_file_tags = DATA_DIR + "/RNN"  +  f"/label_data{suffix}.csv"
        if params_dict["no_cov"]:
            csv_file_cov = None
        else:
            csv_file_cov  = DATA_DIR + "/RNN"  +  f"/cov_data{suffix}.csv"
    else:
        raise "invalid simtype"
    
    csv_files = dict()
    csv_files["path"] = csv_file_path
    csv_files["tags"] = csv_file_tags
    csv_files["cov"]  = csv_file_cov
    return csv_files

if __name__ =="__main__":

    gpu_num = 2
    sim_type = "binned"
    tensor_type = "mat"
    no_cov = False
    
    if no_cov:
        simulation_name=f"21_MS_{sim_type}_{tensor_type}_no_cov_RNN"
    else:
        simulation_name=f"21_MS_{sim_type}_{tensor_type}_RNN"


    device = torch.device(f"cuda:{gpu_num}")
    torch.cuda.set_device(gpu_num)

    
    train_idx = np.load(DATA_DIR + "/folds/train_idx_0.npy", allow_pickle = True)
    val_idx = np.load(DATA_DIR + "/folds/val_idx_0.npy", allow_pickle = True)
    test_idx = np.load(DATA_DIR + "/folds/test_idx_0.npy", allow_pickle  = True)

    #Model parameters.
    params_dict=dict()
    params_dict["hidden_size"] = 50
    
    if sim_type == "binned":
        params_dict["delta_t"]=0.1
    elif tensor_type == "continuous":
        params_dict["delta_t"]=1
    else:
        raise "Incorrect sim-type"

    params_dict["T"]=36


    params_dict["classification_hidden"] = 30
    params_dict["cov_hidden"] = 25
    params_dict["num_layers"] = 1 
    params_dict["weight_decay"] = 0.00005
    params_dict["dropout"] = 0.
    params_dict["lr"]=0.005
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = no_cov
    params_dict["residual_classifier"] = True

    csv_files = files_linking(sim_type, tensor_type, params_dict)

    info, val_metric_prev, test_auc = train_rnn_ms(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        csv_files = csv_files,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=10)

