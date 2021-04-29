import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ms.models.gru_ode_model as gru_ode
import ms.RNN.data_utils as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
#from gru_ode import Logger
import os
from ms import DATA_DIR

def train_gruode_ms(simulation_name, params_dict,device, csv_files, train_idx, val_idx, test_idx, epoch_max=40,init_only = False):

    csv_file_path = csv_files["path"]
    csv_file_tags = csv_files["tags"]
    csv_file_cov  = csv_files["cov"]


    N = pd.read_csv(csv_file_tags)["ID"].nunique()

    #if not init_only:
    #    logger = Logger(f'../../Logs/MS/{simulation_name}')

    validation = True
    
    if "cont" in csv_file_path:
        val_options = {"T_val": 1095, "max_val_samples": 1, "T_closest": 1825, "T_val_from" : 1460}
    else:
        val_options = {"T_val": 36, "max_val_samples": 1, "T_closest":60, "T_val_from" : 48}
    
    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags, cov_file= csv_file_cov, idx=train_idx,validation = validation, val_options = val_options, root_dir = "")
    data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=val_idx, validation = validation,
                                        val_options = val_options, root_dir = "")
    data_test   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
                                        cov_file= csv_file_cov, idx=test_idx, validation = validation,
                                        val_options = val_options, root_dir = "")
    
     
    dl   = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=1000,num_workers=4)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(val_idx))
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=len(test_idx))

    params_dict["input_size"]=data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim

    if not init_only:
        np.save(f"./trained_models/{simulation_name}_params.npy",params_dict)

    pos_weight = torch.tensor(((data_train.label_df.shape[0]/data_train.label_df.sum())-1).values,device = device, dtype = torch.float)
    print(f"Pos weight : {pos_weight}")
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum',pos_weight = pos_weight)
   
    nnfwobj = gru_ode.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                            p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                            logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                            classification_hidden=params_dict["classification_hidden"],
                                            cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                            dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"])
    nnfwobj.to(device)

    if init_only:
        return nnfwobj,class_criterion, dl_test, device, params_dict

    optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay= params_dict["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor = 0.5,patience = 3,verbose = True, min_lr = 1e-7)
    print("Start Training")
    val_metric_prev = -1000
    for epoch in range(epoch_max):
        nnfwobj.train()
        total_train_loss = 0
        auc_total_train  = 0
        class_train_loss = 0
        for i, b in enumerate(tqdm.tqdm(dl)):

            optimizer.zero_grad()
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels   = b["y"].to(device)
            batch_size = labels.size(0)


            h0 = 0# torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, _  = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov)


            total_loss = loss/M.sum() + params_dict["lambda"]*class_criterion(class_pred, labels)/cov.size(0)
            

            total_train_loss += total_loss

            auc_total_train += roc_auc_score(labels.detach().cpu(),torch.sigmoid(class_pred).detach().cpu())

            total_loss.backward()
            optimizer.step()

            class_train_loss += class_criterion(class_pred,labels).detach().cpu()/cov.size(0)
        
        info = { 'training_loss' : total_train_loss.detach().cpu().numpy()/(i+1), "classification_loss": class_train_loss,'AUC_training' : auc_total_train/(i+1), 'learning_rate': optimizer.param_groups[0]["lr"]}
        #for tag, value in info.items():
        #    logger.scalar_summary(tag, value, epoch)

        data_utils.adjust_learning_rate(optimizer,epoch,params_dict["lr"])

        with torch.no_grad():

            loss_val, auc_total_val, mse_val, total_loss_val, corr_val, num_obs_val = test_evaluation(nnfwobj, params_dict, class_criterion, device, dl_val)

            info = { 'validation_loss' : total_loss_val, 'AUC_validation' : auc_total_val,
                     'loglik_loss' : loss_val, 'validation_mse' : mse_val, 'correlation_mean' : np.nanmean(corr_val),
                    'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
            #logger.save_dict(info,epoch)

            if params_dict["lambda"]==0:
                val_metric = - loss_val
            else:
                val_metric = auc_total_val

            if val_metric > val_metric_prev:
                print(f"New highest validation metric reached ! : {val_metric}")
                print("Saving Model")
                torch.save(nnfwobj.state_dict(),f"./trained_models/{simulation_name}_MAX.pt")
                val_metric_prev = val_metric
                test_loglik, test_auc, test_mse, total_loss_test, corr_test, _ = test_evaluation(nnfwobj, params_dict, class_criterion, device, dl_test)
                print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
                print(f"Test AUC loss at epoch {epoch} : {test_auc}")
                print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            else:
                if epoch % 10:
                    torch.save(nnfwobj.state_dict(),f"./trained_models/{simulation_name}.pt")
        
            info_test = {'AUC_test':test_auc}
            #logger.save_dict(info_test,epoch)

        scheduler.step(val_metric)
        
        print(f"Train AUC at epoch {epoch} : {auc_total_train/(i+1)} -  Classification loss : {class_train_loss/(i+1)}")
        print(f"Total validation loss at epoch {epoch}: {total_loss_val}")
        print(f"Validation AUC at epoch {epoch}: {auc_total_val}")
        print(f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs_val}")

    print(f"Finished training GRU-ODE for MS. Saved in ./trained_models/{simulation_name}")

    df_file_name = "./trained_models/MS_results.csv"
    df_res = pd.DataFrame({"Name" : [simulation_name], "AUC_val" : [val_metric_prev], "AUC_test" : [test_auc]})
    if os.path.isfile(df_file_name):
        df = pd.read_csv(df_file_name)
        df = df.append(df_res)
        df.to_csv(df_file_name,index=False)
    else:
        df_res.to_csv(df_file_name,index=False)
    
    return(info, val_metric_prev, test_loglik, test_auc, test_mse)

def test_evaluation(model, params_dict, class_criterion, device, dl_test, post_eval = False):
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        corr_test = 0
        num_obs = 0
        for i, b in enumerate(dl_test):
            times    = b["times"]
            time_ptr = b["time_ptr"]
            X        = b["X"].to(device)
            M        = b["M"].to(device)
            obs_idx  = b["obs_idx"]
            cov      = b["cov"].to(device)
            labels   = b["y"].to(device)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(device)
                M_val     = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]
                X_last    = b["X_last"].to(device)

            h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _  = model(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=True)
            total_loss = loss/M.sum() + params_dict["lambda"]*class_criterion(class_pred, labels)/cov.size(0)

            auc_test=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())

            if params_dict["lambda"]==0:
                t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                m, v = torch.chunk(p_val,2,dim=1)
                m_diff = m-X_last
                #Aggravation criterion. 
                delta_edss =  (X_last<=5.5).float()*1 + (X_last>5.5).float()*0.5

                prob_degradation = data_utils.tail_fun_gaussian(delta_edss.cpu(),m_diff.cpu(),v.cpu())
                auc_test = roc_auc_score(labels.cpu()[times_idx],prob_degradation)

                last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs=1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /=  num_obs
        total_loss_test /= (i+1)
        auc_total_test /= (i+1)

        if post_eval:
            return auc_total_test, labels.cpu(), torch.sigmoid(class_pred).cpu()

        return(loss_test, auc_total_test, mse_test, total_loss_test, corr_test, num_obs)


def files_linking(sim_type, tensor_type, params_dict, suffix = ""):
    if sim_type == "continuous": 
        csv_file_path = DATA_DIR + f"/RNN/{tensor_type}_data_cont{suffix}.csv"
        csv_file_tags = DATA_DIR + f"/RNN/label_data{suffix}.csv"
        if params_dict["no_cov"]:
            csv_file_cov = None
        else:
            csv_file_cov  = DATA_DIR + f"/RNN/cov_data{suffix}.csv"
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

    #gpu_num = 1
    sim_type = "binned"
    tensor_type = "mat"
    no_cov = False
    
    if no_cov:
        simulation_name=f"MS_{sim_type}_{tensor_type}_no_cov"
    else:
        simulation_name=f"MS_{sim_type}_{tensor_type}"


    device = torch.device(f"cuda")
    #torch.cuda.set_device(gpu_num)

    
    train_idx = np.load("/home/edward/Projects/MS/ms/folds/train_idx_0.npy", allow_pickle = True)
    val_idx = np.load("/home/edward/Projects/MS/ms/folds/val_idx_0.npy", allow_pickle = True)
    test_idx = np.load("/home/edward/Projects/MS/ms/folds/test_idx_0.npy", allow_pickle  = True)

    #Model parameters.
    params_dict=dict()
    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 10
    params_dict["prep_hidden"] = 4
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
    
    if sim_type == "binned":
        params_dict["delta_t"]=0.1
    elif tensor_type == "continuous":
        params_dict["delta_t"]=1
    else:
        raise "Incorrect sim-type"

    params_dict["T"]=36
    params_dict["lambda"] = 10000000 #Weighting between classification and MSE loss.

    params_dict["classification_hidden"] = 30
    params_dict["cov_hidden"] = 25
    params_dict["weight_decay"] = 0.0000005
    params_dict["dropout_rate"] = 0.
    params_dict["lr"]=0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = no_cov

    csv_files = files_linking(sim_type, tensor_type, params_dict)
    if no_cov:
        csv_files["cov"] = "./sub_cov_data.csv"

    info, val_metric_prev, test_loglik, test_auc, test_mse = train_gruode_ms(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        csv_files = csv_files,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=100)

