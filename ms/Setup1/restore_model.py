from joblib import load
import numpy as np
import pandas as pd
import ms
import time
from joblib import dump, load
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from ms import DATA_DIR

type_model = "bptf" #bptf, static, dynamic
mscourse = "SP"

infile_path = DATA_DIR + "/" 
Y = pd.read_csv(infile_path+"label_data.csv")

Num_patients= Y.values.shape[0]

X = pd.read_csv(infile_path+"cov_data.csv")

if type_model=="static":
    cov_list = ["gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS",  'last_dmt_mild',
       'last_dmt_moderate', 'last_dmt_high', 'last_dmt_none']
elif type_model =="dynamic":
    cov_list = ["gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","mean_edss","max_previous_EDSS",'Relapses_in_observation_period', 'number_of_visits', "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]
elif type_model =="bptf":
    cov_list = ["gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","mean_edss","max_previous_EDSS",'Relapses_in_observation_period', 'number_of_visits', "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]
else:
    raise("Model type not supported")

X = X[["UNIQUE_ID"]+cov_list]
print(f"Running Model 0 (RF) with covariates :")
print(X.columns)
print(f"{X.shape[0]} patients included")


RF_AUC = np.zeros(5)
RF_FPR = []
RF_TPR = []
RF_precision = []
RF_recall = []
times = []

for fold in range(5):

    XY = X.merge(Y, on = "UNIQUE_ID")
    test_idx = np.load(DATA_DIR + f"/folds/test_idx_{fold}.npy", allow_pickle = True)
   
    if type_model != "bptf":
        if mscourse is not None:
            XY = XY.loc[X[mscourse]>0]

    
        X_test  = XY.loc[XY.UNIQUE_ID.isin(test_idx),cov_list].values
        Y_test  = XY.loc[XY.UNIQUE_ID.isin(test_idx),"confirmed_label"].values
    
    else:
        X_df_test = X.loc[X.UNIQUE_ID.isin(test_idx)]
        if mscourse is not None:
            mask = (X_df_test[mscourse]>0).values
        
        X_test = np.load(f"./saved_models/bptf_X_test_fold{fold}.npy")
        Y_test = np.load(f"./saved_models/bptf_Y_test_fold{fold}.npy")
        if mscourse is not None:
            X_test = X_test[mask]
            Y_test = Y_test[mask]
    

    clf = load(f"./saved_models/{type_model}_model_fold{fold}.joblib")

    
    fpr, tpr,_ = roc_curve(Y_test,clf.predict_proba(X_test)[:,1])
    precision, recall, _ = precision_recall_curve(Y_test,clf.predict_proba(X_test)[:,1])
    auc=roc_auc_score(Y_test,clf.predict_proba(X_test)[:,1])

    RF_AUC[fold] = auc
    RF_FPR.append(fpr)
    RF_TPR.append(tpr) 
    RF_precision.append(precision)
    RF_recall.append(recall)

if mscourse is not None:
    mscourse_str = f"_{mscourse}"
else:
    mscourse_str = ""

model_keys = {"static":0, "dynamic":1, "bptf":6}

np.save(f"../comparisons_results/Setup1/model{model_keys[type_model]}{mscourse_str}.npy",RF_AUC)
np.save(f"../comparisons_results/Setup1/model{model_keys[type_model]}{mscourse_str}_fpr.npy",np.array(RF_FPR))
np.save(f"../comparisons_results/Setup1/model{model_keys[type_model]}{mscourse_str}_tpr.npy",np.array(RF_TPR))
np.save(f"../comparisons_results/Setup1/model{model_keys[type_model]}{mscourse_str}_precision.npy",np.array(RF_precision))
np.save(f"../comparisons_results/Setup1/model{model_keys[type_model]}{mscourse_str}_recall.npy",np.array(RF_recall))

