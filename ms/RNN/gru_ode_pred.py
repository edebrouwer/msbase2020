from class_prop import class_prob, files_linking
import torch
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import time
from ms import DATA_DIR


mscourse = ["PP","PR"] #PP, SP, RR, PR
GRUODE_tpr = []
GRUODE_fpr = []
GRUODE_precision = []
GRUODE_recall = []
GRUODE_auc = []

times = []

for fold in range(5):

    start_time = time.time()
    simulation_name = np.load("./trained_models/best_GRU_ODE_models.npy",allow_pickle = True)[fold]
    device = torch.device("cuda")
    train_idx = np.load(DATA_DIR + f"/folds/train_idx_{fold}.npy",allow_pickle = True)
    val_idx = np.load(DATA_DIR + f"/folds/val_idx_{fold}.npy", allow_pickle = True)
    test_idx = np.load(DATA_DIR + f"/folds/test_idx_{fold}.npy", allow_pickle = True)
   
    gpu_num = 1
    sim_type = "binned"
    tensor_type = "mat"

    device = torch.device(f"cuda:{gpu_num}")
    device = torch.device("cpu")
    torch.cuda.set_device(gpu_num)

    params_dict = np.load(f"./trained_models/{simulation_name}_params.npy", allow_pickle = True).item()

    train_idx = np.load(DATA_DIR + f"/folds/train_idx_{fold}.npy", allow_pickle = True)
    val_idx = np.load(DATA_DIR + f"/folds/val_idx_{fold}.npy", allow_pickle = True)
    test_idx = np.load(DATA_DIR + f"/folds/test_idx_{fold}.npy", allow_pickle  = True)

    #Model parameters.
    if "original" in simulation_name:
        suffix= "_original"
    else:
        suffix = ""

    csv_files = files_linking(sim_type, tensor_type, params_dict, suffix = suffix)

    #csv_files["cov"] = "./sub_cov_data.csv"

    batch_size = len(test_idx)



    label_test, pred_test = class_prob(simulation_name = simulation_name,
                        device = device,
                        csv_files = csv_files,
                        test_idx = test_idx,
                        epoch_max=100, batch_size = batch_size,logits = True, mscourse = mscourse)
    
    fpr, tpr, _  = roc_curve(label_test,pred_test)
    precision, recall, _ = precision_recall_curve(label_test, pred_test)

    auc = roc_auc_score(label_test,pred_test)

    GRUODE_auc.append(auc)
    GRUODE_tpr.append(tpr)
    GRUODE_fpr.append(fpr)
    GRUODE_precision.append(precision)
    GRUODE_recall.append(recall)

    end_time = time.time()
    times.append((end_time-start_time)/len(test_idx))

if mscourse is not None:
    if isinstance(mscourse,list):
        mscourse_str = "_" + "_".join(mscourse)
    else:
        mscourse_str = f"_{mscourse}"
else:
    mscourse_str = ""

np.save(f"./trained_models/GRUODE{mscourse_str}_tpr.npy",np.array(GRUODE_tpr))
np.save(f"./trained_models/GRUODE{mscourse_str}_fpr.npy",np.array(GRUODE_fpr))
np.save(f"./trained_models/GRUODE{mscourse_str}_precision.npy",np.array(GRUODE_precision))
np.save(f"./trained_models/GRUODE{mscourse_str}_recall.npy",np.array(GRUODE_recall))
np.save(f"./trained_models/GRUODE{mscourse_str}_auc.npy",np.array(GRUODE_auc))

