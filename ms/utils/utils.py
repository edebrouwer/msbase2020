import numpy as np
import sys
from sklearn.decomposition import PCA

import pandas as pd
import torch
import sys

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import shutil
import os

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

def train_test_model(L2_param,latent_path,tags,train_idx,test_idx):
    device=torch.device("cuda:0")
    latents_train,latents_test=PCA_macau_samples(dir_path=latent_path,idx_train=train_idx,idx_val=test_idx)
    data_test=latent_dataset(latents_test,tags[test_idx])
    data_train=latent_dataset(latents_train,tags[train_idx])
    mod=MLP_class_mod(data_train.get_dim())
    dataloader=DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)

    criterion=nn.BCEWithLogitsLoss()
    for epoch in range(100):
        if epoch<40:
            l_r=0.01
        elif epoch<60:
            l_r=0.005
        else:
            l_r=0.0005
        optimizer=torch.optim.Adam(mod.parameters(),lr=l_r,weight_decay=L2_param)
        loss=0
        for idx,sampled_batch in enumerate(dataloader):
            optimizer.zero_grad()
            target=sampled_batch[1]
            preds=mod.fwd(sampled_batch[0])
            loss=criterion(preds,target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            target=data_test.tags
            preds=F.sigmoid(mod.fwd(data_test.latents))
            loss_test=roc_auc_score(target,preds)

        print(f"Loss on test is  : {loss_test}")

    torch.save(mod.state_dict(),"complete_model.pt")

def get_best_params(path):
    dir_path="/home/edward/ray_results/"+path
    dirs= os.listdir(dir_path)
    for i,dir in enumerate(dirs):
        file_name=os.path.join(os.path.join(dir_path,dir),"progress.csv")
        df=pd.read_csv(file_name)
        if i==0:
            results=df.tail(1)
        else:
            results=results.append(df.tail(1))

    best_run=results.loc[results["mean_accuracy"]==results["mean_accuracy"].max()]

    best_config=eval(best_run["config"].iloc[0])

    best_L2=best_config["L2"]
    return(best_L2)


def PCA_macau_samples(dir_path,idx_train=None,idx_val=None,num_samples=50,n_dim=None,fold=None):
    sum_sim=np.load(dir_path+"sum_sim.npy").item()
    N_latents=sum_sim["N_latents"]
    N_samples=sum_sim["N_samples"]

    if fold is not None:
        prefix=f"_macau_fold_{fold}"
    else:
        prefix="_macau"

    concat_lat=np.loadtxt(dir_path+str(N_latents)+prefix+"-sample1-U1-latents.csv",delimiter=",")

    print("Concat sample")
    for n in np.linspace(10,N_samples,num_samples,dtype='int'):
        concat_lat=np.concatenate((concat_lat,np.loadtxt(dir_path+str(N_latents)+prefix+"-sample%d-U1-latents.csv"%n,delimiter=",")))
    print("Done")
    if idx_train is not None:
        concat_subset=concat_lat[:,idx_train]
    else:
        concat_subset=concat_lat

    pca=PCA()
    pca.fit(concat_subset.T)
    #print(np.cumsum(pca.explained_variance_ratio_))

    if n_dim is  None:
        n_kept=np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.9))
    else:
        n_kept=n_dim
    pca=PCA(n_components=n_kept)
    pca.fit(concat_subset.T)

    pca_latents=pca.transform(concat_lat.T)

    np.save(dir_path+"pca_latents",pca_latents)
    print(n_kept)

    return(pca_latents[idx_train,:],pca_latents[idx_val,:])

def box_plots_comparisons(AUC_vecs,names):
    data=np.vstack(AUC_vecs)
    fig, ax = plt.subplots()
    ax.boxplot(data.T)
    plt.title("AUC Performance Comparison for EDSS worsening")
    ax.set_xticklabels(names)
    plt.savefig("./comparisons_results/box_plot.pdf")

class MLP_class_mod(nn.Module):
    def __init__(self,input_dim):
        super(MLP_class_mod,self).__init__()
        self.layer_1=nn.Linear(input_dim,100)
        #self.layer_1bis=nn.Linear(200,70)
        self.layer_2=nn.Linear(100,20)
        self.layer_3=nn.Linear(20,1)
    def fwd(self,x):
        out=F.relu(self.layer_1(x))
        #out=F.relu(self.layer_1bis(out))
        out=F.relu(self.layer_2(out))
        out=self.layer_3(out).squeeze(1)
        return(out)

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
