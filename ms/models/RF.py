import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import optunity
import optunity.metrics


from ms.utils.data_utils import extract_covariates



def RF_pred(X_train,Y_train, X_test, Y_test,n_splits=5, num_evals = 100):

    @optunity.cross_validated(x=X_train,y=Y_train,num_folds=n_splits)
    def fun_max(x_train,y_train,x_test,y_test,n_estimators, max_depth, max_features, min_samples_split):
        
        if max_features<0.5:
            max_features = "auto"
        else:
            max_features = "sqrt"

        clf=RandomForestClassifier(n_estimators=int(n_estimators),max_depth=int(max_depth),
                                    max_features = max_features, min_samples_split = int(min_samples_split), class_weight="balanced")
        clf.fit(x_train,y_train)
        score=optunity.metrics.roc_auc(y_test,clf.predict_proba(x_test)[:,1])

        #print(f"Average AUC on {n_splits}-fold validation with {int(n_estimators)} trees of max depth :{max_depth} = {score}")
        return(score)

    #Grid Search definition.
    # Number of trees in random forest
    n_estimators = [100,1000]
    # Number of features to consider at every split
    max_features = [0, 1]
    # Maximum number of levels in tree
    max_depth = [5,25]
    #max_depth.append(None)
    #Minimum number of samples required to split a node
    min_samples_split = [2,10]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1,4]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'max_features': max_features,
                   'min_samples_split': min_samples_split
                   #'min_samples_leaf': min_samples_leaf}
                   #,
                  #'bootstrap': bootstrap
                   }

    optimal_parameters, info, _ = optunity.maximize_structured(fun_max,search_space=random_grid,num_evals=num_evals,pmap=optunity.pmap)
    print(f"Optimal parameters : {optimal_parameters} with AUC of {info.optimum}")

    #Evaluating on test set :
    if optimal_parameters["max_features"]<0.5:
        max_feats = "auto"
    else:
        max_feats = "sqrt"

    clf=RandomForestClassifier(n_estimators=int(optimal_parameters["n_estimators"]),
            max_depth=int(optimal_parameters["max_depth"]), 
                max_features = max_feats,
                min_samples_split = int(optimal_parameters["min_samples_split"]),class_weight="balanced")
    clf.fit(X_train,Y_train)
    fpr, tpr,_ = roc_curve(Y_test,clf.predict_proba(X_test)[:,1])
    precision, recall, _ = precision_recall_curve(Y_test,clf.predict_proba(X_test)[:,1])
    #np.save("./plots/fpr_RF.npy",fpr)
    #np.save("./plots/tpr_RF.npy",tpr)
    score=roc_auc_score(Y_test,clf.predict_proba(X_test)[:,1])
    print(f"ROC AUC with optimal set of hyperparameters : {score}")
    return(score,fpr, tpr, precision, recall,clf)

if __name__=="__main__":
    infile_path="~/Data/MS/"
    Num_patients=pd.read_csv(infile_path+"Label_bis_data.csv").values.shape[0]
    train_idx, test_idx = train_test_split(np.arange(Num_patients),test_size=0.1)
    RF_pred_next_EDSS(train_idx,test_idx)
