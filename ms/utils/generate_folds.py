import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ms import DATA_DIR

unique_ids = pd.read_csv(DATA_DIR + "/label_data.csv").UNIQUE_ID.unique()
Num_patients = len(unique_ids)
seed = 421

for sim in range(5):
    train_idx, test_idx = train_test_split(np.arange(Num_patients),test_size=0.15,random_state=seed+sim)
    train_idx, val_idx  = train_test_split(train_idx, test_size = 0.2, random_state = seed+sim)
    
    train_idx = unique_ids[train_idx]
    val_idx   = unique_ids[val_idx]
    test_idx  = unique_ids[test_idx]

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0

    np.save(DATA_DIR + f"/folds/train_idx_{sim}.npy",train_idx)
    np.save(DATA_DIR + f"/folds/val_idx_{sim}.npy",val_idx)
    np.save(DATA_DIR + f"/folds/test_idx_{sim}.npy",test_idx)



