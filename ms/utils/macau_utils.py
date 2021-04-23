import macau
import pandas as pd
import numpy as np
from scipy.stats import norm
import tqdm

def get_macau_pred(eval_idx=None, allowed_idx=None, num_folds=1, from_onset=False, Setup2 = False, tensor_mode = False, covariates_list = None, get_latents = False):
    """
    eval_idx : idx for which an prob evaluation is required
    allowed_idx : idx which can be used for training
    """

    N=500

    infile_path = "~/Data/MS/Cleaned_MSBASE/"
    outfile_path = "./macau_sim/"

    if from_onset:
        Y = pd.read_csv(infile_path+"mat_from_onset_data.csv")[["UNIQUE_ID","Binned_time_from_onset","EDSS","Binned_shift"]]
        Y.rename(columns={"Binned_time_from_onset":"Binned_time"},inplace=True)
    elif Setup2:
        Y = pd.read_csv(infile_path+"mat_Setup2_data.csv")[["UNIQUE_ID","Binned_time_from_onset","EDSS","Binned_shift"]]
        Y.rename(columns={"Binned_time_from_onset":"Binned_time"},inplace=True)
    elif tensor_mode:
        Y = pd.read_csv(infile_path+"tens_data.csv")
    else:
        Y = pd.read_csv(infile_path+"mat_data.csv")[["UNIQUE_ID","Binned_time","EDSS"]]
        Y = Y.loc[~Y.EDSS.isna()].copy()

    if eval_idx is None:
        eval_idx = Y["UNIQUE_ID"].unique()
    if allowed_idx is None:
        allowed_idx = Y["UNIQUE_ID"].unique()

    Y = Y.loc[Y["UNIQUE_ID"].isin(allowed_idx)].copy()

    idx_map = dict(zip(allowed_idx,np.arange(len(allowed_idx))))
    inv_map = {v: k for k,v in idx_map.items()} 
    
    if Setup2:
        last_obs = Y.sort_values(by=["UNIQUE_ID","Binned_time"]).drop_duplicates(subset=["UNIQUE_ID"],keep="first")[["UNIQUE_ID","Binned_time","EDSS","Binned_shift"]]
        other_obs = Y[~Y.isin(last_obs)].dropna()
        other_obs["UNIQUE_ID"] = other_obs["UNIQUE_ID"].astype(int)
        other_obs["Binned_time"] = other_obs["Binned_time"].astype(int)
        other_obs["Binned_shift"] = other_obs["Binned_shift"].astype(int)
    elif from_onset:
        Y["Binned_time_from_1st"] = Y["Binned_time"]-Y["Binned_shift"].astype(int)
        Y_prevs = Y.loc[Y["Binned_time_from_1st"]<37].sort_values(by=["UNIQUE_ID","Binned_time"]).copy()
        last_obs=Y_prevs.drop_duplicates(subset=["UNIQUE_ID"],keep="last")[["UNIQUE_ID","Binned_time","EDSS","Binned_shift"]]
    elif tensor_mode:
        Y_edss = Y.loc[Y["LABEL"]==0].copy()
        Y_prevs=Y_edss.loc[Y_edss["Binned_time"]<37].sort_values(by=["UNIQUE_ID","Binned_time"]).copy()
        last_obs=Y_prevs.drop_duplicates(subset=["UNIQUE_ID"],keep="last")[["UNIQUE_ID","VALUE"]]
    else: #Data is aligned on the first visit time.
        Y_prevs=Y.loc[Y["Binned_time"]<37].sort_values(by=["UNIQUE_ID","Binned_time"]).copy()
        last_obs=Y_prevs.drop_duplicates(subset=["UNIQUE_ID"],keep="last")[["UNIQUE_ID","EDSS"]]

    total_preds = pd.DataFrame(columns = ["UNIQUE_ID","PROBABILITY"])
    total_preds["UNIQUE_ID"] = total_preds["UNIQUE_ID"].astype(int)
    latents_df = pd.DataFrame()
    num_latents = 70

    if covariates_list is not None:
        if Setup2:
            X=pd.read_csv(infile_path+"cov_Setup2_data.csv")
        else:
            X = pd.read_csv(infile_path+"cov_data.csv")
        cov2 = X.loc[X["UNIQUE_ID"].isin(allowed_idx)]
        cov2["UNIQUE_ID"] = cov2["UNIQUE_ID"].map(idx_map)
        cov2.sort_values(by = "UNIQUE_ID", inplace = True)
        cov = cov2[covariates_list].values

    else:
        cov = None

    for pat_idx in tqdm.tqdm(np.array_split(eval_idx,num_folds)):
        train_idx = np.setdiff1d(allowed_idx,pat_idx)
        test_idx  = pat_idx
        assert(len(np.intersect1d(train_idx,test_idx))==0)
        
        #We select the values in the tensor that correspond to the training patients and crop the ones of test patients.
        if Setup2:
            Y_train = Y.loc[Y["UNIQUE_ID"].isin(train_idx)].append(last_obs.loc[last_obs["UNIQUE_ID"].isin(test_idx)])[["UNIQUE_ID","Binned_time","EDSS"]]
            Y_test = other_obs.loc[other_obs["UNIQUE_ID"].isin(test_idx),["UNIQUE_ID","Binned_time","EDSS"]]
            
            #We only care about prediction at binned time  = 365*5//30, so we add this one :
            eval_df = last_obs.loc[last_obs["UNIQUE_ID"].isin(test_idx)].copy()
            eval_df["Binned_time"]=eval_df["Binned_shift"]+((365*5)//30)*np.ones(test_idx.shape[0]).astype(int)
            eval_df["EDSS"]=0
            eval_df.drop(columns = "Binned_shift",inplace=True)
        
            Y_test = Y_test.append(eval_df)
        elif from_onset:
            Y_train=Y.loc[(Y["Binned_time_from_1st"]<=((365*3)//30)) | (Y["UNIQUE_ID"].isin(train_idx))][["UNIQUE_ID","Binned_time","EDSS"]]

            Y_test=Y.loc[(Y["Binned_time_from_1st"]>((365*3)//30)) & (Y["UNIQUE_ID"].isin(test_idx))][["UNIQUE_ID","Binned_time","EDSS"]]
    
            
            eval_df = last_obs.loc[last_obs["UNIQUE_ID"].isin(test_idx)].copy()
            eval_df["Binned_time"]=eval_df["Binned_shift"]+((365*5)//30)*np.ones(test_idx.shape[0]).astype(int)
            eval_df["EDSS"]=0
            eval_df.drop(columns = "Binned_shift",inplace=True)
            Y_test = Y_test.append(eval_df)

        elif tensor_mode:
            Y_train=Y.loc[(Y["Binned_time"]<=((365*3)//30)) | (Y["UNIQUE_ID"].isin(train_idx))]

            Y_test=Y.loc[(Y["Binned_time"]>((365*3)//30)) & (Y["UNIQUE_ID"].isin(test_idx))]

            Y_test = Y_test.loc[Y["LABEL"]==0].copy()
   
            #We only care about prediction at binned time  = 365*5//30, so we add this one : 
            Y_at_eval_time = pd.DataFrame({ 'UNIQUE_ID' : pat_idx,
                                        'LABEL': 0,
                                        'Binned_time' : ((365*5)//30)*np.ones(pat_idx.shape[0]).astype(int), 
                                        'VALUE' : np.zeros(pat_idx.shape[0])})

            Y_test = Y_test.append(Y_at_eval_time)



        else:
            Y_train=Y.loc[(Y["Binned_time"]<=((365*3)//30)) | (Y["UNIQUE_ID"].isin(train_idx))]

            Y_test=Y.loc[(Y["Binned_time"]>((365*3)//30)) & (Y["UNIQUE_ID"].isin(test_idx))]
    
            
            #We only care about prediction at binned time  = 365*5//30, so we add this one : 
            Y_at_eval_time = pd.DataFrame({ 'UNIQUE_ID' : pat_idx,
                                        'Binned_time' : ((365*5)//30)*np.ones(pat_idx.shape[0]).astype(int), 
                                        'EDSS' : np.zeros(pat_idx.shape[0]) })

            Y_test = Y_test.append(Y_at_eval_time)

        Y_train["UNIQUE_ID"] = Y_train["UNIQUE_ID"].map(idx_map)
        Y_test["UNIQUE_ID"]  = Y_test["UNIQUE_ID"].map(idx_map)


        if get_latents:
            results=macau.macau(Y=Y_train,Ytest=Y_test,side=[cov,None],num_latent=num_latents,burnin=100,nsamples=N, verbose = False, save_prefix = "MacauTemp")
            latents = np.zeros((Y_train["UNIQUE_ID"].nunique(),num_latents))
            for sim in range(N):
                latents += np.loadtxt(f"../Setup1/MacauTemp-sample{sim+1}-U1-latents.csv",delimiter = ",").transpose()
            latents_df_temp = pd.DataFrame(latents/N)
            latents_df_temp["UNIQUE_ID"] = Y_train["UNIQUE_ID"].unique()
            latents_df_temp = latents_df_temp.loc[latents_df_temp["UNIQUE_ID"].isin(test_idx)].copy()
            latents_df = latents_df.append(latents_df_temp)
        else:
            if tensor_mode:
                side = [cov,None,None]
            else:
                side = [cov,None]
           
            #Y_train = Y_train[["UNIQUE_ID","Binned_time","VALUE"]]
            #Y_test  = Y_test[["UNIQUE_ID","Binned_time","VALUE"]]
            #side = [cov,None]
            results=macau.macau(Y=Y_train,Ytest=Y_test,side=side,num_latent=num_latents,burnin=400,nsamples=N, verbose = False)
        preds=results.prediction
        preds["UNIQUE_ID"] = preds.UNIQUE_ID.map(inv_map)

        if Setup2:
            preds = eval_df[["UNIQUE_ID","Binned_time"]].merge(preds,on=["UNIQUE_ID","Binned_time"])
        elif from_onset:
            preds = eval_df[["UNIQUE_ID","Binned_time"]].merge(preds,on=["UNIQUE_ID","Binned_time"])
        else:
            preds = preds.loc[preds["Binned_time"]==(365*5)//30].copy()
       
        if tensor_mode:
            preds=preds.merge(last_obs[["UNIQUE_ID","VALUE"]],on="UNIQUE_ID").copy().rename(columns={"VALUE":"last_observed_edss"})
        else:
            preds=preds.merge(last_obs[["UNIQUE_ID","EDSS"]],on="UNIQUE_ID").copy().rename(columns={"EDSS":"last_observed_edss"})

        preds["prob_label1"]=norm.sf(preds["last_observed_edss"]+1,loc=preds["y_pred"],scale=preds["y_pred_std"])
        preds["prob_label2"]=norm.sf(preds["last_observed_edss"]+0.5,loc=preds["y_pred"],scale=preds["y_pred_std"])

        preds["prev>5.5"] = preds["last_observed_edss"]>5.5
        preds["prev<=5.5"] = preds["last_observed_edss"]<=5.5

        preds["prob_label_medic"] = preds["prev>5.5"]*preds["prob_label1"] + preds["prev<=5.5"]*preds["prob_label2"]

        pred_summary = preds[["UNIQUE_ID","prob_label_medic"]].rename(columns={"prob_label_medic":"PROBABILITY"})
        total_preds = total_preds.append(pred_summary).reset_index(drop=True)
    
    if get_latents:
        return(total_preds, latents_df)
    else:
        return(total_preds)
        
if __name__ == "__main__":
    eval_idx = np.load("../folds/test_idx_0.npy")
    preds = get_macau_pred(eval_idx = eval_idx, num_folds = 1, tensor_mode = True)  
