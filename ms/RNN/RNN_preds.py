from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import time
from ms.models.RNN import evaluate_model, RNN_classification_model, files_linking
import torch
import numpy as np
import ms.RNN.data_utils as data_utils
from torch.utils.data import DataLoader
import time
from ms import DATA_DIR

def get_preds(simulation_name, device, csv_files, train_idx, val_idx, test_idx, mscourse = None):
    csv_file_path = csv_files["path"]
    csv_file_tags = csv_files["tags"]
    csv_file_cov  = csv_files["cov"]


    params_dict = np.load(f"./trained_models/{simulation_name}_params.npy",allow_pickle = True).item()

    N = pd.read_csv(csv_file_tags)["ID"].nunique()


    validation = True
    
    if "cont" in csv_file_path:
        val_options = {"T_val": 1095, "max_val_samples": 1, "T_closest": 1825, "T_val_from" : 1460}
    else:
        val_options = {"T_val": 36, "max_val_samples": 1, "T_closest":60, "T_val_from" : 48}
    
    data_train   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=train_idx, validation = validation,
                                        val_options = val_options)

    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=val_idx, validation = validation,
                                        val_options = val_options)
    data_test   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options)
   
    dl_train = DataLoader(dataset=data_train, collate_fn=data_utils.discrete_collate_fn, shuffle=False, batch_size=len(train_idx))
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.discrete_collate_fn, shuffle=False, batch_size=len(val_idx))
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.discrete_collate_fn, shuffle=False, batch_size=len(test_idx))

    model_name = simulation_name

    ## model definition
    model = RNN_classification_model(params_dict)
    model.to(device)

    model.load_state_dict(torch.load(f"./trained_models/{model_name}_best_val.pt"))

    auc_train, complication_pred_train, complication_label_train, train_indices, auc_pr_train = evaluate_model(model,data_train, dl_train, device)
    auc_val, complication_pred_val, complication_label_val, val_indices, auc_pr_val  = evaluate_model(model,data_val,dl_val, device)
    start_time = time.time()
    auc_test, complication_pred_test, complication_label_test, test_indices, auc_pr_test  = evaluate_model(model,data_test, dl_test, device)
    end_time = time.time()

    if mscourse is not None:
        ms_course_index_train =  np.array(data_train.cov_df.loc[(data_train.cov_df[mscourse]>0).any(1)].index.tolist()).astype(int)
        ms_course_index_val =  np.array(data_val.cov_df.loc[(data_val.cov_df[mscourse]>0).any(1)].index.tolist()).astype(int)
        ms_course_index_test =  np.array(data_test.cov_df.loc[(data_test.cov_df[mscourse]>0).any(1)].index.tolist()).astype(int)

        complication_pred_train  = complication_pred_train[ms_course_index_train]
        complication_label_train = complication_label_train[ms_course_index_train]

        complication_pred_val  = complication_pred_val[ms_course_index_val]
        complication_label_val = complication_label_val[ms_course_index_val]
        
        complication_pred_test  = complication_pred_test[ms_course_index_test]
        complication_label_test = complication_label_test[ms_course_index_test]
    
    train_id_map = {v:k for k,v in data_train.map_dict.items()}
    val_id_map = {v:k for k,v in data_val.map_dict.items()}
    test_id_map = {v:k for k,v in data_test.map_dict.items()}

    train_ids = np.vectorize(train_id_map.get)(train_indices)
    val_ids = np.vectorize(val_id_map.get)(val_indices)
    test_ids = np.vectorize(test_id_map.get)(test_indices)
    return (train_ids, complication_pred_train,complication_label_train), (val_ids,complication_pred_val,complication_label_val),  (test_ids,complication_pred_test,complication_label_test) , (end_time-start_time)/len(test_idx)

if __name__ =="__main__":
    
    #None, PP, PR, RR, SP
    mscourse = ["PP","PR"]
    GRU_tpr = []
    GRU_fpr = []
    GRU_precision = []
    GRU_recall = []

    times = []
    for fold in range(5):
        simulation_name = np.load( "./trained_models/best_RNN_models.npy",allow_pickle = True)[fold]
        device = torch.device("cuda")
        train_idx = np.load(DATA_DIR + f"/folds/train_idx_{fold}.npy",allow_pickle = True)
        val_idx = np.load(DATA_DIR + f"/folds/val_idx_{fold}.npy", allow_pickle = True)
        test_idx = np.load(DATA_DIR + f"/folds/test_idx_{fold}.npy", allow_pickle = True)

        sim_type = "binned"
        tensor_type = "mat"
        suffix = "_original"
        params_dict = np.load(f"./trained_models/{simulation_name}_params.npy",allow_pickle = True).item()

        csv_files = files_linking(sim_type, tensor_type, params_dict, suffix = suffix)
        label_df = pd.read_csv(csv_files["tags"])
    
        (train_ids, pred_train,label_train), (val_ids,pred_val,label_val), (test_ids,pred_test, label_test), computation_time = get_preds(simulation_name, device, csv_files, train_idx, val_idx, test_idx, mscourse = mscourse)
        times.append(computation_time)

        preds = np.concatenate((pred_train,pred_val,pred_test))
        labels = np.concatenate((label_train,label_val,label_test))

        fpr, tpr, _  = roc_curve(label_test,pred_test)
        precision, recall, _ = precision_recall_curve(label_test, pred_test)


        GRU_tpr.append(tpr)
        GRU_fpr.append(fpr)
        GRU_precision.append(precision)
        GRU_recall.append(recall)

        if mscourse is None:
            label_train_o = label_df.set_index("ID").loc[train_ids].reset_index()
            label_train_o["pred"] = pred_train
            label_train_o.to_csv(f"./trained_models/preds_RNN/fold{fold}/train_preds.csv",index= False)
            label_val_o = label_df.set_index("ID").loc[val_ids].reset_index()
            label_val_o["pred"] = pred_val
            label_val_o.to_csv(f"./trained_models/preds_RNN/fold{fold}/val_preds.csv",index= False)
            label_test_o = label_df.set_index("ID").loc[test_ids].reset_index()
            label_test_o["pred"] = pred_test
            label_test_o.to_csv(f"./trained_models/preds_RNN/fold{fold}/test_preds.csv",index= False)
    
    if mscourse is not None:
        if isinstance(mscourse,list):
            mscourse_str = "_"+"_".join(mscourse)
        else:
            mscourse_str = f"_{mscourse}"
    else:
        mscourse_str = ""
    np.save(f"./trained_models/GRU{mscourse_str}_tpr.npy",np.array(GRU_tpr))
    np.save(f"./trained_models/GRU{mscourse_str}_fpr.npy",np.array(GRU_fpr))
    np.save(f"./trained_models/GRU{mscourse_str}_precision.npy",np.array(GRU_precision))
    np.save(f"./trained_models/GRU{mscourse_str}_recall.npy",np.array(GRU_recall))

    mean_time = np.array(times).mean()
    std_time  = np.array(times).std()
    
