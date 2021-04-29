import pandas as pd
from ms import DATA_DIR

#Process the EDSS data
file_path = DATA_DIR + "/"
outpath = DATA_DIR + "/RNN/"
label_name= "confirmed_label" #label_ter
df = pd.read_csv(file_path+"extended_mat_data.csv")

for col in ['Relapse_count_binned','DMT_START_MILD_COUNTS', 'DMT_START_MODERATE_COUNTS', 'DMT_START_HIGH_COUNTS', 'DMT_END_MILD_COUNTS','DMT_END_MODERATE_COUNTS', 'DMT_END_HIGH_COUNTS']:
    df.loc[df[col].isna(),col] = 0.

df.rename(columns = {"UNIQUE_ID": "ID","Binned_time":"Time","EDSS":"Value_0",'Relapse_count_binned':"Value_1",'DMT_START_MILD_COUNTS':'Value_2', 'DMT_START_MODERATE_COUNTS':'Value_3', 'DMT_START_HIGH_COUNTS':'Value_4', 'DMT_END_MILD_COUNTS':'Value_5','DMT_END_MODERATE_COUNTS': 'Value_6', 'DMT_END_HIGH_COUNTS':'Value_7'},inplace = True)


for col in df.columns:
    if "Value" in col:
        idx = col.split("_")[1]
        df[f"Mask_{idx}"] = (~df[col].isna()).astype(int)

#Normalize only the EDSS.
df["Value_0"] = (df["Value_0"]-df["Value_0"].mean())/df["Value_0"].std()

df.fillna(0,inplace = True)

df.to_csv(outpath+"mat_data.csv", index = False)


#Process the covariates data
df_cov = pd.read_csv(file_path+"cov_data.csv")
df_cov.rename(columns = {"UNIQUE_ID":"ID"}, inplace = True)
covariates_list = ["ID","gender_class","PP","PR","RR","SP","DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed","last_EDSS","first_EDSS","diff_EDSS","max_previous_EDSS","mean_edss",
        'Relapses_in_observation_period', "number_of_visits", "last_dmt_mild","last_dmt_moderate","last_dmt_high","last_dmt_none"]

df_cov = df_cov[covariates_list]
#Normalize data
df_cov.iloc[:,1:] = (df_cov.iloc[:,1:]-df_cov.iloc[:,1:].mean())/df_cov.iloc[:,1:].std()

df_cov.to_csv(outpath+"cov_data.csv",index = False)

#Process the label data
df_lab = pd.read_csv(file_path+"label_data.csv")[["UNIQUE_ID", label_name]]
df_lab.rename(columns = {"UNIQUE_ID":"ID",label_name:"label"},inplace = True)
df_lab.to_csv(outpath+"label_data.csv",index = False)


