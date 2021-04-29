import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

name_base = "./trained_models/Xval_GRUODE_fold"
test_auc = []
sim_names = []
for i in range(5):
    print(f"fold {i} is computing ...")
    df = pd.read_csv(name_base+str(i)+".csv")
    df = df.loc[df.simulation_name.str.contains("21_")]
    if df.shape[0] == 0:
        continue
    best_row = df.loc[df.Best_val_AUC.idxmax()]
    test_auc += [best_row["Best_test_AUC"]]
    sim_names += [best_row['simulation_name']]

test_auc = np.array(test_auc)
print(f"Test AUC averaged : {test_auc.mean()} +- {test_auc.std()}")
np.save("./trained_models/best_GRU_ODE_models.npy",sim_names)

df = pd.read_csv("./trained_models/MS_results.csv")

auc_test = np.array([ df.iloc[df.loc[(df.Name.str.contains("fold"+str(i)) & df.Name.str.contains("continuous"))].AUC_val.idxmax()].AUC_test for i in range(5)])
auc_test.mean()
auc_test.std()

name_base = "./trained_models/21_Xval_RNN_fold"
test_auc_rnn = []
best_models_names = []
for i in range(5):
    print(f"Computing fold {i}")
    df = pd.read_csv(name_base+str(i)+".csv")
    df = df.loc[df.simulation_name.str.contains("21_original_residual_Xval")]
    if df.shape[0]==0:
        continue
    best_row = df.loc[df.Best_val_AUC.idxmax()]
    best_models_names.append(best_row.simulation_name)
    test_auc_rnn += [best_row["Best_test_AUC"]]

test_auc_rnn = np.array(test_auc_rnn)
np.save("./trained_models/best_RNN_models.npy",best_models_names)
print(f"Test AUC RNN averaged : {test_auc_rnn.mean()} +- {test_auc_rnn.std()}")
np.save("./trained_models/GRU_auc.npy",test_auc_rnn)



