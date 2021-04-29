import pandas as pd
from ms import DATA_DIR

out_suffix = "_original"

#Process the EDSS data
file_path = DATA_DIR + "/"
outpath = DATA_DIR + "/RNN/"

label_name= "confirmed_label" #label_ter
df = pd.read_csv(file_path+"mat_data.csv")
df = df.loc[~df.EDSS.isna()].copy()

df.rename(columns = {"UNIQUE_ID": "ID","Binned_time":"Time","EDSS":"Value_0"},inplace = True)
df["Mask_0"]=1


df["Value_0"] = (df["Value_0"]-df["Value_0"].mean())/df["Value_0"].std()

df.to_csv(outpath+f"mat_data{out_suffix}.csv", index = False)


#Process the covariates data
df_cov = pd.read_csv(file_path+"cov_data.csv")
df_cov.rename(columns = {"UNIQUE_ID":"ID"}, inplace = True)
#covariates_list = ["ID","gender_class","CIS","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","max_previous_EDSS",
#        'Relapses_in_observation_period', "number_of_visits", 'Interferons', 'other', 'Alemtuzumab','Glatiramer', 'Natalizumab', 'Fingolimod', 'Teriflunomide', 'Cladribine', 'no_dmt_found',
#       'Dimethyl-Fumarate', 'Ocrelizumab', 'Rituximab']
covariates_list = ["ID","gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","max_previous_EDSS","mean_edss",
        'Relapses_in_observation_period', "number_of_visits", "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]

df_cov = df_cov[covariates_list]
#Normalize data
df_cov.iloc[:,1:] = (df_cov.iloc[:,1:]-df_cov.iloc[:,1:].mean())/df_cov.iloc[:,1:].std()

df_cov.to_csv(outpath + f"cov_data{out_suffix}.csv",index = False)

#Process the label data
df_lab = pd.read_csv(file_path+"label_data.csv")[["UNIQUE_ID", label_name]]
df_lab.rename(columns = {"UNIQUE_ID":"ID",label_name:"label"},inplace = True)
df_lab.to_csv(outpath + f"label_data{out_suffix}.csv",index = False)


        
