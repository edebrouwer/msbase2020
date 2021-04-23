"""
This is the static model ! Using random forest

"""

import numpy as np
import pandas as pd
import ms
import time
from joblib import dump, load

infile_path = "~/Data/MS/Cleaned_MSBASE/"
Y = pd.read_csv(infile_path+"label_data.csv")

Num_patients= Y.values.shape[0]

X = pd.read_csv(infile_path+"cov_data.csv")
cov_list = ["gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS",  'last_dmt_mild',
       'last_dmt_moderate', 'last_dmt_high', 'last_dmt_none']

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
    train_idx = np.load(f"../folds/train_idx_{fold}.npy", allow_pickle = True)
    val_idx  = np.load(f"../folds/val_idx_{fold}.npy", allow_pickle =True)
    test_idx = np.load(f"../folds/test_idx_{fold}.npy", allow_pickle = True)

    train_idx = np.concatenate((train_idx,val_idx))

    XY = X.merge(Y, on = "UNIQUE_ID")
    X_train = XY.loc[XY.UNIQUE_ID.isin(train_idx),cov_list].values
    X_test  = XY.loc[XY.UNIQUE_ID.isin(test_idx),cov_list].values
    #Y_train = XY.loc[XY.UNIQUE_ID.isin(train_idx),"label_ter"].values
    #Y_test  = XY.loc[XY.UNIQUE_ID.isin(test_idx),"label_ter"].values
    Y_train = XY.loc[XY.UNIQUE_ID.isin(train_idx),"confirmed_label"].values
    Y_test  = XY.loc[XY.UNIQUE_ID.isin(test_idx),"confirmed_label"].values

    auc, fpr, tpr, precision, recall, clf = ms.models.RF.RF_pred(X_train,Y_train,X_test,Y_test)
    RF_AUC[fold] = auc
    RF_FPR.append(fpr)
    RF_TPR.append(tpr) 
    RF_precision.append(precision)
    RF_recall.append(recall)
    
    start_time = time.time()
    clf.predict(X_test)
    end_time = time.time()

    times.append((end_time-start_time)/len(test_idx))

    dump(clf,f"./saved_models/static_model_fold{fold}.joblib")

np.save("../comparisons_results/Setup1/model0.npy",RF_AUC)
np.save("../comparisons_results/Setup1/model0_fpr.npy",np.array(RF_FPR))
np.save("../comparisons_results/Setup1/model0_tpr.npy",np.array(RF_TPR))
np.save("../comparisons_results/Setup1/model0_precision.npy",np.array(RF_precision))
np.save("../comparisons_results/Setup1/model0_recall.npy",np.array(RF_recall))

print(f"Average AUC :{np.mean(RF_AUC)}")
