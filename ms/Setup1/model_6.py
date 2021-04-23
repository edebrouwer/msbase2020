import numpy as np
import pandas as pd
import ms
from ms.utils.macau_utils import get_macau_pred
import time

from joblib import dump, load

infile_path = "~/Data/MS/Cleaned_MSBASE/"
Y = pd.read_csv(infile_path+"label_data.csv")

Num_patients= Y.values.shape[0]

X = pd.read_csv(infile_path+"cov_data.csv")
covariates_list = ["PROBABILITY","gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","mean_edss","max_previous_EDSS",'Relapses_in_observation_period', 'number_of_visits', "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]

macau_covariates_list = ["gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","max_previous_EDSS","mean_edss",'Relapses_in_observation_period', 'number_of_visits', "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]
 

print(f"Running Model 6 (RF + Macau) with covariates :")
print(covariates_list)
print(f"Side information : ")
print(macau_covariates_list)

Model6_AUC = np.zeros(5)
Model6_FPR = []
Model6_TPR = []
Model6_precision = []
Model6_recall = []

times = []
for fold in range(5):
    print(f"Computing BPMF for fold {fold}...")
    train_idx =np.load(f"../folds/train_idx_{fold}.npy", allow_pickle = True)
    test_idx = np.load(f"../folds/test_idx_{fold}.npy",allow_pickle = True)

    train_macau_preds = get_macau_pred(eval_idx = train_idx, allowed_idx = train_idx, num_folds=10,covariates_list = macau_covariates_list)

    start_time = time.time()
    test_macau_preds = get_macau_pred(eval_idx = test_idx, num_folds = 1, covariates_list = macau_covariates_list)
    end_time = time.time()

    times.append((end_time-start_time)/len(test_idx))

    macau_prob = train_macau_preds.append(test_macau_preds)
    X_full = X.merge(macau_prob,on="UNIQUE_ID")

    X_train = X_full.loc[X_full.UNIQUE_ID.isin(train_idx),covariates_list].values
    Y_train = Y.loc[Y.UNIQUE_ID.isin(train_idx),"confirmed_label"].values
    X_test  = X_full.loc[X_full.UNIQUE_ID.isin(test_idx),covariates_list].values
    Y_test  = Y.loc[Y.UNIQUE_ID.isin(test_idx),"confirmed_label"].values

    
    print(f"BPMF done.")
    print(f"Computing RF model...")
    auc, fpr, tpr, precision, recall, clf = ms.models.RF.RF_pred(X_train,Y_train,X_test,Y_test)
    
    Model6_AUC[fold] = auc
    Model6_FPR.append(fpr)
    Model6_TPR.append(tpr)
    Model6_precision.append(precision)
    Model6_recall.append(recall)
    
    dump(clf,f"./saved_models/bptf_model_fold{fold}.joblib")
    np.save(f"./saved_models/bptf_X_test_fold{fold}.npy",X_test)
    np.save(f"./saved_models/bptf_Y_test_fold{fold}.npy",Y_test)


np.save("../comparisons_results/model6.npy",Model6_AUC)
np.save("../comparisons_results/model6_tpr.npy",np.array(Model6_TPR))
np.save("../comparisons_results/model6_fpr.npy",np.array(Model6_FPR))
np.save("../comparisons_results/model6_precision.npy",np.array(Model6_precision))
np.save("../comparisons_results/model6_recall.npy",np.array(Model6_recall))


print(f"Average AUC :{np.mean(Model6_AUC)} +- {np.std(Model6_AUC)}")

