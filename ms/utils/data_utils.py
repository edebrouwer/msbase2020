import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#import torch
#from torch.utils.data import Dataset, DataLoader

def extract_covariates(df,static_only=False):
    df_clean=df.sort_values(by="UNIQUE_ID")
    X=df_clean.drop_duplicates(subset=["UNIQUE_ID"],keep="first")[["UNIQUE_ID","BIRTH_DATE","first_edss","first_visit_date","Time_first2last","gender_class","max_previous_edss","previous_edss","CIS","PP","PR","RR","SP","onset_date"]]
    X["duration"]=X["Time_first2last"].dt.days/365
    X["Age_at_onset"]=(X["onset_date"]-X["BIRTH_DATE"]).dt.days/365
    X["duration_at_T0"]=(X["first_visit_date"]-X["onset_date"]).dt.days/365
    X["edss_diff"]=X["previous_edss"]-X["first_edss"]
    #X=X[["edss_diff","duration","gender_class",'max_previous_edss',"Age_at_onset","duration_at_T0","CIS","PP","PR","RR","SP"]].values
    if static_only:
        X=X[["previous_edss","gender_class","CIS","PP","PR","RR","SP"]].values
    else:
        X=X[["edss_diff","gender_class",'max_previous_edss',"CIS","PP","PR","RR","SP"]].values
    return(X)

def train_val_test_split(X,sizes=(0.2,0.1)):
    train_val_idx, test_idx=train_test_split(X,test_size=sizes[1])
    t_idx, v_idx = train_test_split(np.arange(train_val_idx.shape[0]),test_size=sizes[0])
    train_idx=train_val_idx[t_idx]
    val_idx=train_val_idx[v_idx]
    return(train_idx, val_idx, test_idx)

"""
class latent_and_cov_dataset(Dataset):
    def __init__(self,latents,tags,covs):
        self.lats=torch.Tensor(latents)
        self.covs=torch.Tensor(covs)
        print(self.covs[:2,:])
        self.latents=torch.cat((self.lats,self.covs),1)
        self.tags=torch.from_numpy(tags).float()
    def __len__(self):
        return(self.latents.size(0))
    def __getitem__(self,idx):
        return([self.latents[idx,:],self.tags[idx]])
    def get_dim(self):
        return self.latents.size(1)

class latent_dataset(Dataset):
    def __init__(self,latents,tags):
        self.latents=torch.Tensor(latents)
        self.tags=torch.from_numpy(tags).float()
    def __len__(self):
        return(self.latents.size(0))
    def __getitem__(self,idx):
        return([self.latents[idx,:],self.tags[idx]])
    def get_dim(self):
        return self.latents.size(1)

"""
