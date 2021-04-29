import numpy as np
import pandas as pd
import ms
from ms.utils.macau_utils import get_macau_pred

infile_path = "~/Data/MS/Cleaned_MSBASE/"
Y = pd.read_csv(infile_path+"label_data.csv")

Num_patients= Y.values.shape[0]

X = pd.read_csv(infile_path+"cov_data.csv")
covariates_list = ["PROBABILITY","gender_class","CIS","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","max_previous_EDSS",'Relapses_in_observation_period', 'number_of_visits', 'Interferons','Glatiramer', 'Alemtuzumab', 'other', 'Fingolimod', 'Natalizumab','no_dmt_found', 'Teriflunomide', 'Cladribine', 'Dimethyl-Fumarate',
'Ocrelizumab', 'Rituximab']

 

print(f"Running Model 6 BIS (RF + Macau) with covariates (no SI):")
print(covariates_list)
print(f"Side information : ")
print(macau_covariates_list)

Model6_AUC = np.zeros(5)
Model6_FPR = []
Model6_TPR = []
Model6_precision = []
Model6_recall = []
for fold in range(5):
    print(f"Computing BPMF for fold {fold}...")
    train_idx =np.load(f"../folds/train_idx_{fold}.npy")
    test_idx = np.load(f"../folds/test_idx_{fold}.npy")

    train_macau_preds = get_macau_pred(eval_idx = train_idx, allowed_idx = train_idx, num_folds=10)
    test_macau_preds = get_macau_pred(eval_idx = test_idx, num_folds = 1)

    macau_prob = train_macau_preds.append(test_macau_preds)
    X_full = X.merge(macau_prob,on="UNIQUE_ID")

    X_train = X_full.loc[X_full.UNIQUE_ID.isin(train_idx),covariates_list].values
    Y_train = Y.loc[Y.UNIQUE_ID.isin(train_idx),"confirmed_label"].values
    X_test  = X_full.loc[X_full.UNIQUE_ID.isin(test_idx),covariates_list].values
    Y_test  = Y.loc[Y.UNIQUE_ID.isin(test_idx),"confirmed_label"].values

    
    print(f"BPMF done.")
    print(f"Computing RF model...")
    auc, fpr, tpr, precision, recall = ms.models.RF.RF_pred(X_train,Y_train,X_test,Y_test)
    
    Model6_AUC[fold] = auc
    Model6_FPR.append(fpr)
    Model6_TPR.append(tpr)
    Model6_precision.append(precision)
    Model6_recall.append(recall)
    
np.save("../comparisons_results/model6_bis.npy",Model6_AUC)
np.save("../comparisons_results/model6_bis_tpr.npy",np.array(Model6_TPR))
np.save("../comparisons_results/model6_bis_fpr.npy",np.array(Model6_FPR))
np.save("../comparisons_results/model6_bis_precision.npy",np.array(Model6_precision))
np.save("../comparisons_results/model6_bis_recall.npy",np.array(Model6_recall))


print(f"Average AUC :{np.mean(Model6_AUC)} +- {np.std(Model6_AUC)}")

