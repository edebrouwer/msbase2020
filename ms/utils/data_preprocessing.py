import pandas as pd
import numpy as np

class printing_class:
    def __init__(self):
        self.prev_total_visits=0
        self.prev_total_patients=0
    def printing_patients_entries(self,df):
        print("Number of patients remaining : {}. Diff = {}".format(df["PATIENT_ID"].nunique(),df["PATIENT_ID"].nunique()-self.prev_total_patients))
        print("Total number of visits : {}. Diff= {}".format(len(df.index),len(df.index)-self.prev_total_visits))
        self.prev_total_visits=len(df.index)
        self.prev_total_patients=df["PATIENT_ID"].nunique()


def process_patients(patients_df = "~/Data/MS/Cleaned_MSBASE/patients.csv"):

    df=pd.read_csv(patients_df,encoding='latin1')
    #Check that every patient has a onset date
    assert(sum(df["ms_onset_reformat"].isnull()==False)==len(df.index))
    #Check that every patient has an age at first symptoms
    assert(sum(df["age_at_first_symptoms_days"].isnull()==False)==len(df.index))
    #Average age at first symptoms.
    mean_age_1symptom = df["age_at_first_symptoms_days"].mean()/365
    print(f"Average age at first symptoms : {mean_age_1symptom}")


    #Date time conversions.
    df["BIRTH_DATE"]=pd.to_datetime(df["BIRTH_DATE"])
    df["onset_date"]=pd.to_datetime(df["ms_onset_reformat"],format="%d%b%Y")
    df['START_OF_PROGRESSION']=pd.to_datetime(df['START_OF_PROGRESSION'])
    df["SYMPTOMS_DATE"]=pd.to_datetime(df['SYMPTOMS_DATE'])
    df["MS_DIAGNOSIS_DATE_reformat"]=pd.to_datetime(df['MS_DIAGNOSIS_DATE'],errors="coerce")
    df["CLINIC_ENTRY_DATE"]=pd.to_datetime(df['CLINIC_ENTRY_DATE'],errors="coerce")
    df["DATE_OF_FIRST_RELAPSE"]=pd.to_datetime(df['DATE_OF_FIRST_RELAPSE'],errors="coerce")
       
    onset_age=df["onset_date"]-df["BIRTH_DATE"]
    df["age_at_onset_days_computed"]=onset_age.dt.days
            
    #Delete patients with invalid diagnosis date.
    df=df.drop(df.loc[(df["MS_DIAGNOSIS_DATE_reformat"].isnull() & df["MS_DIAGNOSIS_DATE"].notnull())].index).copy()
    print("Number of remaining patients : {}".format(df["PATIENT_ID"].nunique()))

    from datetime import datetime
    today=datetime(2019,1,1)

    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["MS_DIAGNOSIS_DATE_reformat"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["SYMPTOMS_DATE"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["BIRTH_DATE"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["onset_date"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["START_OF_PROGRESSION"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["CLINIC_ENTRY_DATE"]>today].index)))
    print("Number of patients with dates > {} : {}".format(today,len(df.loc[df["DATE_OF_FIRST_RELAPSE"]>today].index)))


    df=df.drop(df.loc[df["MS_DIAGNOSIS_DATE_reformat"]>today].index).copy()
    df=df.drop(df.loc[df["SYMPTOMS_DATE"]>today].index).copy()
    df=df.drop(df.loc[df["BIRTH_DATE"]>today].index).copy()
    df=df.drop(df.loc[df["onset_date"]>today].index).copy()
    df=df.drop(df.loc[df["START_OF_PROGRESSION"]>today].index).copy()
    df=df.drop(df.loc[df["CLINIC_ENTRY_DATE"]>today].index).copy()
    df=df.drop(df.loc[df["DATE_OF_FIRST_RELAPSE"]>today].index).copy()

    #Todo : remove patients that are too similar (might be the same patients)

    df_subset=df[['PATIENT_ID', 'BIRTH_DATE', 'gender', 'CLINIC_ENTRY_DATE', 'DATE_OF_FIRST_RELAPSE',
                      'SYMPTOMS_DATE', 'START_OF_PROGRESSION', 
                      'PROGRESSION_FROM_ONSET',
                      'MS_DIAGNOSIS_DATE_reformat', 
                      'onset_date',
                      'age_at_first_symptoms_days',
                      'clinic_code', 'age_at_onset_days_computed']].copy()
    df_subset.rename({'MS_DIAGNOSIS_DATE_reformat':'MS_DIAGNOSIS_DATE'},inplace=True,axis='columns')
    df_subset.to_pickle("/home/edward/Data/MS/Cleaned_MSBASE/patients_clean.pkl")
    print("Patients file processed. Saved in ~/Data/MS/Cleaned_MSBASE/patients_clean.pkl")


def create_MRI_df(visits_df, MRI_csv = "~/Data/MS/Cleaned_MSBASE/MRI.csv",ratio = 30, end_test = 6):
    df = pd.read_csv(MRI_csv)
    patients_df = visits_df.drop_duplicates("PATIENT_ID",keep="first")
    patient_clean_idx = patients_df["PATIENT_ID"]
    df=df.loc[df["PATIENT_ID"].isin(patient_clean_idx)].copy()
    #Clean NAs
    df=df.drop(df.loc[(df["T1_GADOLINIUM_LESION"].isnull())&(df["T2_LESION"].isnull())].index).copy()
    # Clean values which are >15 or >10 std.(see below thresholds)
    max_gad = df["T1_GADOLINIUM_LESION"].std()*15
    max_T2  = df["T2_LESION"].std()*10

    df.loc[df["T1_GADOLINIUM_LESION"]>max_gad,"T1_GADOLINIUM_LESION"]=np.nan
    df.loc[df["T2_LESION"]>max_T2,"T2_LESION"]=np.nan

    df["EXAM_DATE"] = pd.to_datetime(df["EXAM_DATE"],errors="coerce")
    
    df = df.merge(patients_df[["PATIENT_ID","first_visit_date","UNIQUE_ID"]],on="PATIENT_ID").copy()

    df["Time"]=df["EXAM_DATE"]-df["first_visit_date"]
    df["Binned_time"] = df["Time"].dt.days//ratio

    #Make a thin dataset.
    df_thin_T1 = df[["UNIQUE_ID","Binned_time","T1_GADOLINIUM_LESION","Time"]].dropna().copy()
    df_thin_T2 = df[["UNIQUE_ID","Binned_time","T2_LESION","Time"]].dropna().copy()
    df_thin_T1["LABEL"]="T1"
    df_thin_T2["LABEL"]="T2"
    df_thin = df_thin_T1.rename(columns={"T1_GADOLINIUM_LESION":"VALUE"}).append(df_thin_T2.rename(columns={"T2_LESION":"VALUE"}))

    #Select only times in the interval [0,6*365//ratio]
    end_test_binned = end_test*365//ratio
    df_thin = df_thin.loc[(df_thin["Binned_time"]>=0)&(df_thin["Binned_time"]<end_test_binned)].copy()

    assert df_thin.VALUE.isna().sum()==0

    return(df_thin)

def process_visits(visits_df = "~/Data/MS/Cleaned_MSBASE/visits.csv",clean_pats_pkl= "/home/edward/Data/MS/Cleaned_MSBASE/patients_clean.pkl",ratio=30):

    print_instance=printing_class()
    cols_to_use = ['PATIENT_ID', 'VISIT_FK', 'DATE_OF_VISIT', 'EDSS',
                    'KFS_1', 'KFS_2', 'KFS_3', 'KFS_4', 'KFS_5', 'KFS_6',
                    'KFS_7','KFS_AMBULATION', 'DURATION_OF_MS_AT_VISIT',
                    'DURATION_OF_MS_AT_VISIT_ROUNDED','FIRST_MSCOURSE',
                    'MSCOURSE_AT_VISIT']
    df=pd.read_csv(visits_df,usecols = cols_to_use)
    patients_df=pd.read_pickle(clean_pats_pkl)
    print_instance.printing_patients_entries(df)
    

    #Linking with patients table.

    patient_clean_idx=patients_df["PATIENT_ID"]
    df=df.loc[df["PATIENT_ID"].isin(patient_clean_idx)].copy()

    df_link=patients_df.merge(df,on="PATIENT_ID")
    df=df_link.copy()

    print_instance.printing_patients_entries(df_link)

            #Basic Cleaning of the data.
            #Remove entries with missing EDSS
    df=df.drop(df.loc[df["EDSS"].isnull()].index).copy()
    print_instance.printing_patients_entries(df)

            #Remove inconsistent visits (same day but with different EDSS measurements)
    df1=df.drop_duplicates(subset=["PATIENT_ID","DATE_OF_VISIT"],keep=False) #contains only entries wich are not duplicated

    several_visits=df.duplicated(subset=["PATIENT_ID","DATE_OF_VISIT"],keep=False) 
    df2=df[several_visits] # contains all duplicated entries
    assert((len(df1.index)+len(df2.index))==len(df.index))

            #Remove the entries with same date and patients but different EDSS.
    df3=df2.drop_duplicates(subset=["PATIENT_ID","DATE_OF_VISIT","EDSS"],keep="first")
    df4=df3.drop_duplicates(subset=["PATIENT_ID","DATE_OF_VISIT"],keep=False)

    df=df1.append(df4)
    assert((len(df1.index)+len(df4.index))==len(df.index))
    assert(sum(df.duplicated(subset=["PATIENT_ID","DATE_OF_VISIT"],keep=False))==0)
    print_instance.printing_patients_entries(df)

            #Create date_visits_processed (conversion to datetime)
    df["DATE_OF_VISIT"]=pd.to_datetime(df["DATE_OF_VISIT"],errors="coerce")
    df=df.drop(df.loc[df["DATE_OF_VISIT"].isnull()].index).copy() #Drop wrong dates

    #Remove entries whose date > 2019.
    from datetime import datetime
    today=datetime(2019,1,1)
    df=df.drop(df.loc[(df["DATE_OF_VISIT"]>today)].index).copy()

    print_instance.printing_patients_entries(df)

            #Remove entries where onset date > fist visit: 
    df=df.drop(df.loc[df["onset_date"]>df["DATE_OF_VISIT"]].index).copy()
    assert(len(df.loc[df["onset_date"]>df["DATE_OF_VISIT"]].index)==0)

            #Create new columns
    """
            Time 
            first_visit_date   
            last_visit_date  
            first_EDSS : EDSS at the first visit  
            last_EDSS : EDSS at the last visit  
            Time_first2last : time between first and last visit  
            lowest_EDSS : lowest EDSS in the patient trajectory  
            highest_EDSS : highest ....  
            diff_EDSS : difference between highest and lowest EDSS over the full trajectory of the patient 
    """
    df_ascend=df.sort_values(by=["PATIENT_ID","DATE_OF_VISIT"])

    df_first=df_ascend.drop_duplicates(subset=["PATIENT_ID"],keep="first")[["PATIENT_ID","EDSS","DATE_OF_VISIT"]]
    df_first.rename(columns={'EDSS':'first_EDSS',"DATE_OF_VISIT":"first_visit_date"},inplace=True)

    df_merged=df.merge(df_first,on="PATIENT_ID")
            #-------Analysis of the number of hits in function of the binning.
    df_merged["Time"]=df_merged["DATE_OF_VISIT"]-df_merged["first_visit_date"]
    df_merged["Time_from_onset"]=df_merged["DATE_OF_VISIT"]-df_merged["onset_date"]
    df_merged["Binned_time"]=df_merged["Time"].dt.days//ratio
    df_merged["Binned_time_from_onset"]=df_merged["Time_from_onset"].dt.days//ratio
    lost_percent=sum(df_merged.duplicated(subset=["Binned_time","PATIENT_ID"],keep="first"))/len(df_merged.index)
    print("By choosing a binning of ratio {}, we loose {} % of the data".format(ratio,lost_percent))
    df_merged=df_merged.drop_duplicates(subset=["Binned_time","PATIENT_ID"],keep="first").copy()
            #-------------------

    df_ascend=df_merged.sort_values(by=["PATIENT_ID","DATE_OF_VISIT"])
    df_last=df_ascend.drop_duplicates(subset=["PATIENT_ID"],keep="last")[["PATIENT_ID","EDSS","DATE_OF_VISIT"]]
    df_last.rename(columns={'EDSS':'last_EDSS',"DATE_OF_VISIT":"last_visit_date"},inplace=True)

    assert(sum(df_first.duplicated(subset=["PATIENT_ID"]))==0)
    assert(sum(df_last.duplicated(subset=["PATIENT_ID"]))==0)


    df_merged=df_merged.merge(df_last,on="PATIENT_ID")

    df_merged["last_visit_date"]=pd.to_datetime(df_merged["last_visit_date"])
    df_merged["first_visit_date"]=pd.to_datetime(df_merged["first_visit_date"])

    df_merged["Time_first2last"]=df_merged["last_visit_date"]-df_merged["first_visit_date"]

    df_merged=df_merged[df_merged.duplicated(subset=["PATIENT_ID"],keep=False)].copy()

    df_ascend_EDSS=df_merged.sort_values(by=["PATIENT_ID","EDSS"])
    df_first_EDSS=df_ascend_EDSS.drop_duplicates(subset=["PATIENT_ID"],keep="first")[["PATIENT_ID","EDSS"]]
    df_first_EDSS.rename(columns={'EDSS':'lowest_EDSS'},inplace=True)
    df_last_EDSS=df_ascend_EDSS.drop_duplicates(subset=["PATIENT_ID"],keep="last")[["PATIENT_ID","EDSS"]]
    df_last_EDSS.rename(columns={'EDSS':'highest_EDSS'},inplace=True)

    df_merged=df_merged.merge(df_first_EDSS,on="PATIENT_ID")
    df_merged=df_merged.merge(df_last_EDSS,on="PATIENT_ID")

    df_merged["diff_EDSS"]=df_merged["highest_EDSS"]-df_merged["lowest_EDSS"]

    df_merged.sort_values(by=["PATIENT_ID","DATE_OF_VISIT"],inplace=True)


            #Number of visits larger than visit_num
    visit_num=3
    print("Number of patients with number of visits > {}: {}".format(visit_num,sum(df_merged.groupby("PATIENT_ID")["VISIT_FK"].count()>visit_num)))

    visits_df=df_merged.groupby("PATIENT_ID")["DATE_OF_VISIT"].count().reset_index()
    visits_df["VISITS_NUM"]=visits_df["DATE_OF_VISIT"]
    visits_df=visits_df[["PATIENT_ID","VISITS_NUM"]].copy()

    df_merged=df_merged.merge(visits_df,on="PATIENT_ID")


	#Remove patients with low number of visits and with visits before 1990.

    min_visits=3
    min_date=datetime(1990,1,1)

            #First remove patients with visits before 1990.
    df_clean=df_merged.drop(df_merged.loc[df_merged["first_visit_date"]<min_date].index).copy()
    print("Remaining patients in the database : {}".format(df_clean["PATIENT_ID"].nunique()))


            #Second remove patient with less 3 visits.
    visits_df=df_clean.groupby("PATIENT_ID")["DATE_OF_VISIT"].count().reset_index()
    visits_df["VISITS_NUM2"]=visits_df["DATE_OF_VISIT"]
    visits_df=visits_df[["PATIENT_ID","VISITS_NUM2"]].copy()
    df_clean=df_clean.merge(visits_df,on="PATIENT_ID")
    df_clean=df_clean.drop(df_clean.loc[df_clean["VISITS_NUM2"]<min_visits].index).copy()
    print("Remaining patients in the database : {}".format(df_clean["PATIENT_ID"].nunique()))


            #Remove patients with onset_date <1990
    df_clean=df_clean.drop(df_clean.loc[df_clean["onset_date"]<min_date].index).copy()
    print("Remaining patients in the database : {}".format(df_clean["PATIENT_ID"].nunique()))
    
    #rel = pd.read_csv("~/Data/MS/Cleaned_MSBASE/relapses.csv")
    #rel["DATE_OF_ONSET"] = pd.to_datetime(rel["DATE_OF_ONSET"])
    #df_clean["DATE_OF_VISIT"] = pd.to_datetime(df_clean["DATE_OF_VISIT"])
    
    #print("computing last relapse date")
    #import ipdb; ipdb.set_trace()
    #df_clean["last_relapse_date"] = df_clean.apply(lambda x: last_visit_computation(x,rel),axis=1)
    
    df_clean.to_csv("~/Data/MS/Cleaned_MSBASE/df_clean.csv",index = False)
    
    return(df_clean)

def last_visit_computation(x,rel):
    df_ = rel.loc[rel.PATIENT_ID==x["PATIENT_ID"]]["DATE_OF_ONSET"].copy()
    delta = (df_ - x["DATE_OF_VISIT"]).dt.days
    if delta[delta<=0].shape[0]>0:
        return df_.loc[delta[delta<=0].idxmax()]
    else:
        return None


def subset_patients_onset(df,ratio,start_test=4,end_test=6):
    #Only select patients that have at least one observation in the interval 4-6 years after first visit.
    start_test_binned = (start_test*365)//ratio
    end_test_binned   = (end_test*365)//ratio
    df_46=df.loc[(df["Binned_time"]<end_test_binned)&(df["Binned_time"]>start_test_binned)]
    pat_idx=df_46["PATIENT_ID"].unique()
    df=df.loc[df["PATIENT_ID"].isin(pat_idx)].copy()
    print("Number of patients remaining in the dataframe : {}".format(df["PATIENT_ID"].nunique()))

    return(df)

def subset_patients(df,ratio,years_limit=3,min_observed_visits=5,start_test=4,end_test=6, num_edss_in_test_window = 1):
    """
    Subset procedure for the patients aligned on the first visit date.
    """
    print("Subset the patients...")
    end_observation_binned = (years_limit*365)//ratio
    start_test_binned = (start_test*365)//ratio
    end_test_binned = (end_test*365)//ratio
    
    df_sub=df.loc[df["Time_first2last"].dt.days>years_limit*365].copy()
    df_sub["Time_since_1st"]=df_sub["DATE_OF_VISIT"]-df_sub["first_visit_date"]
    print("Number of patients remaining in the dataframe : {}".format(df_sub["PATIENT_ID"].nunique()))

    #Count the number of entries in the time_frame
    df_less=df_sub.loc[df_sub["Time_since_1st"].dt.days<=years_limit*365].copy()
    assert(df_sub["PATIENT_ID"].nunique()==df_less["PATIENT_ID"].nunique())

    visits_df=df_less.groupby("PATIENT_ID")["DATE_OF_VISIT"].count().reset_index()
    patient_index_enough=visits_df.loc[visits_df["DATE_OF_VISIT"]>min_observed_visits]

    df_clean=df_sub.loc[df_sub["PATIENT_ID"].isin(patient_index_enough["PATIENT_ID"])].copy()
    print("Number of patients remaining in the dataframe : {}".format(df_clean["PATIENT_ID"].nunique()))
  
    #first_visit_date_dict 
    first_visit_date_dict = df_clean.groupby("PATIENT_ID")["first_visit_date"].first().to_dict()

    #Process the relapses and add the count in the observation window as a covariate   
    print("Process Relapses...")

    rel = pd.read_csv("~/Data/MS/Cleaned_MSBASE/relapses.csv")
    rel = rel.loc[rel.PATIENT_ID.isin(df_clean.PATIENT_ID.unique())].copy()
    rel["DATE_OF_ONSET"] = pd.to_datetime(rel["DATE_OF_ONSET"],errors="coerce")
    rel["Relapse"] = True
    rel = rel[["PATIENT_ID","DATE_OF_ONSET","Relapse","severity"]]
    rel.dropna("index",subset = ["DATE_OF_ONSET"],inplace = True)


    df_m = pd.merge(df_clean,rel,on = ["PATIENT_ID"], how = "left")
    df_m["time_since_relapse"] = (df_m["DATE_OF_VISIT"]-df_m["DATE_OF_ONSET"]).dt.days
    df_m.loc[df_m.DATE_OF_ONSET.isna(),"time_since_relapse"] = -9999 # no relapse for these patients
    assert df_m.time_since_relapse.isna().sum() == 0
    df_m.loc[df_m["time_since_relapse"]<0,"time_since_relapse"] = 9999
    
    df_m = df_m.sort_values(by=["PATIENT_ID","DATE_OF_VISIT","time_since_relapse"])
    df_m = df_m.drop_duplicates(subset = ["PATIENT_ID","DATE_OF_VISIT"],keep = "first")
    df_m.rename(columns = {"DATE_OF_ONSET":"last_relapse_date"},inplace = True)
    df_m.loc[df_m["time_since_relapse"]==9999,"last_relapse_date"] = None
    df_m.drop(columns = ["severity","Relapse"],inplace = True)

    rel_obs = rel.loc[rel.PATIENT_ID.isin(df_m.PATIENT_ID.unique())].copy()
    rel_obs.rename(columns = {"DATE_OF_ONSET":"DATE_OF_VISIT"}, inplace = True)
    rel_obs = rel_obs.loc[~rel_obs.DATE_OF_VISIT.isna()].copy()
    rel_obs.drop_duplicates(subset = ["PATIENT_ID","DATE_OF_VISIT"], keep = "first", inplace = True)

    df_with_rel = pd.merge(df_m,rel_obs, how = "outer", on = ["PATIENT_ID","DATE_OF_VISIT"])
    df_with_rel.loc[df_with_rel.Relapse.isna(),"Relapse"] = False
    df_with_rel.sort_values(["PATIENT_ID","DATE_OF_VISIT"], inplace = True)
    
    df_with_rel["Relapse_flag"] = df_with_rel["Relapse"].astype(int)
    df_with_rel["cumulative_relapse"] = df_with_rel.groupby("PATIENT_ID")["Relapse_flag"].cumsum()
    df_with_rel.drop(columns = ["Relapse_flag"],inplace = True)

    df_with_rel.loc[df_with_rel.Relapse,"time_since_relapse"] = 0

    assert df_with_rel.time_since_relapse.isna().sum()==0

    df_with_rel["first_visit_date"] = df_with_rel["PATIENT_ID"].map(first_visit_date_dict)
    df_with_rel["Time"] = df_with_rel["DATE_OF_VISIT"]-df_with_rel["first_visit_date"]
    
    df_clean = df_with_rel.copy()

    #Remove the EDSS in the test period that are less than 30 days after a relapse.
    df_4=df_clean.loc[(df_clean["Time"].dt.days>((start_test)*365))&(~df_clean.EDSS.isna())].copy()
    #df_4["time_since_relapse"] = df_4["DATE_OF_VISIT"]-pd.to_datetime(df_4["last_relapse_date"])
    df_clean.drop(df_4.loc[df_4.time_since_relapse<=30].index, inplace = True)

    #Select edss entries with 4y<Time_since_1st<6y
    df_46=df_clean.loc[(df_clean["Time"].dt.days<((end_test)*365))&(df_clean["Time"].dt.days>((start_test)*365))&(~df_clean.EDSS.isna())]
    df_46g = df_46.groupby("PATIENT_ID")["EDSS"].count()
    pat_idx= list(df_46g.loc[df_46g>num_edss_in_test_window].index)
    df_clean=df_clean.loc[df_clean["PATIENT_ID"].isin(pat_idx)].copy()
    print("Number of patients remaining in the dataframe : {}".format(df_clean["PATIENT_ID"].nunique()))
   

    #Process the relapses and add the count in the observation window as a covariate    
    #rel = pd.read_csv("~/Data/MS/Cleaned_MSBASE/relapses.csv")
    #rel_timed = rel[["PATIENT_ID","DATE_OF_ONSET"]].merge((df_sub.groupby("PATIENT_ID")["first_visit_date"].min().reset_index()),how = "outer", on = "PATIENT_ID")
    #rel_timed = rel_timed.loc[rel_timed.PATIENT_ID.isin(df_sub.PATIENT_ID.unique())]
    #rel_timed.dropna(inplace = True)
    #rel_timed.DATE_OF_ONSET = pd.to_datetime(rel_timed.DATE_OF_ONSET)
    #rel_timed["Delta_time"] = (rel_timed["DATE_OF_ONSET"]-rel_timed["first_visit_date"])
    #rel_timed["Binned_time"] = rel_timed["Delta_time"].dt.days//ratio
    #rel_timed = rel_timed.loc[rel_timed.Binned_time<=end_observation_binned]
    #rel_to_merge = rel_timed.groupby("PATIENT_ID")["Binned_time"].count().reset_index().rename(columns = {"Binned_time":"Relapses_in_observation_period"})
    #df_clean = df_clean.merge(rel_to_merge,how = "left", on = "PATIENT_ID")
    
    
    print("Process DMTs....")
    #Process the dmts and add the last one in the observation window as a covariate
    dmt = pd.read_csv("~/Data/MS/Cleaned_MSBASE/treatment_clean.csv")
    dmt = dmt.loc[dmt.PATIENT_ID.isin(df_clean.PATIENT_ID.unique())].copy()

    dmt["START_DATE"] = pd.to_datetime(dmt["START_DATE"],errors="coerce")
    dmt["END_DATE"] = pd.to_datetime(dmt["END_DATE"],errors="coerce")
    
    dmt_start = dmt[["PATIENT_ID","DMT_GROUP","START_DATE"]].copy().rename(columns = {"START_DATE":"DATE_OF_VISIT","DMT_GROUP":"DMT_START"})
    dmt_start["DMT_START_TYPE"] = True  
    df_m = pd.merge(df_clean, dmt_start, how = "outer", on = ["PATIENT_ID","DATE_OF_VISIT"])
    df_m.loc[df_m.DMT_START_TYPE.isna(),"DMT_START_TYPE"] = False

    dmt_end = dmt[["PATIENT_ID","DMT_GROUP","END_DATE"]].copy().rename(columns = {"END_DATE":"DATE_OF_VISIT","DMT_GROUP":"DMT_END"})
    dmt_end["DMT_END_TYPE"] = True  
    df_m = pd.merge(df_m, dmt_end, how = "outer", on = ["PATIENT_ID","DATE_OF_VISIT"])
    df_m.loc[df_m.DMT_END_TYPE.isna(),"DMT_END_TYPE"] = False

    df_m.drop(columns = ["KFS_1","KFS_2","KFS_3","KFS_4","KFS_5","KFS_6","KFS_7","KFS_AMBULATION"], inplace = True)

    df_clean = df_m.copy()

    dmt_story = pd.read_csv("~/Data/MS/Cleaned_MSBASE/treatment_history_clean.csv")
    dmt_story = dmt_story.loc[dmt_story.PATIENT_ID.isin(df_clean.PATIENT_ID.unique())].copy()
    dmt_story["DATE_OF_VISIT"] = pd.to_datetime(dmt_story["date"])
    dmt_story.drop(columns = ["date"],inplace = True)
    df_m = df_clean.merge(dmt_story, how = "left", on = ["PATIENT_ID","DATE_OF_VISIT"])
    df_m.sort_values(by=["PATIENT_ID","DATE_OF_VISIT"],inplace = True)
    df_m["ACTIVE_DMT"] = df_m.groupby("PATIENT_ID")["ACTIVE_DMT"].transform(lambda v : v.ffill())


    df_clean = df_m.copy()


    #dmt = df_clean[["PATIENT_ID","Binned_time"]].merge(dmt,how = "left")
    #dmt = dmt.loc[dmt.Binned_time<=(years_limit*365)//ratio]
    #dmt = dmt.loc[~dmt.dmt_clean.isna()]
    #dmt = dmt.sort_values(["PATIENT_ID","Binned_time"])
    #dmt = dmt.drop_duplicates(subset = ["PATIENT_ID"],keep = "last")
    
    #df_clean = df_clean.merge(dmt[["PATIENT_ID","dmt_clean"]].rename(columns = {"dmt_clean":"last_dmt"}), how = "left", on = "PATIENT_ID")
    #df_clean.loc[df_clean.last_dmt.isna(),"last_dmt"] = "no_dmt_found"
    
    return(df_clean,end_observation_binned,start_test_binned, end_test_binned)

def prepare_dataset(df,end_observation_binned,start_test_binned,end_test_binned,ratio):
    import numpy as np

    #d=dict(zip(list(df["PATIENT_ID"].unique()),np.arange(df["PATIENT_ID"].nunique())))

    df["first_visit_date"] = df["PATIENT_ID"].map(df.groupby("PATIENT_ID")["first_visit_date"].first().to_dict())

    df["Time"]=df["DATE_OF_VISIT"]-df["first_visit_date"]
    df["Binned_time"] = df["Time"].dt.days//ratio
    
    df = df.loc[df.Time.dt.days>=0].copy()

    df.loc[df.Relapse.isna(),"Relapse"] = False
    df.loc[df.DMT_START_TYPE.isna(),"DMT_START_TYPE"] = False
    df.loc[df.DMT_END_TYPE.isna(),"DMT_END_TYPE"] = False
    
    df_edss = df.loc[~df.EDSS.isna()].copy()
    #Aggregating relapses
    df_rel = df.groupby(["PATIENT_ID","Binned_time"])["Relapse"].sum().reset_index().rename(columns = {"Relapse":"Relapse_count_binned"})[["PATIENT_ID","Binned_time","Relapse_count_binned"]]
    df_rel["Relapse_count_binned"] = df_rel["Relapse_count_binned"].astype(int)
   
    df["DMT_START_TYPE"] = df["DMT_START_TYPE"].astype(int)
    df["DMT_END_TYPE"] = df["DMT_END_TYPE"].astype(int)

    df_dmt_start = df.groupby(["PATIENT_ID","Binned_time","DMT_START"])["DMT_START_TYPE"].sum().reset_index().rename(columns = {"DMT_START_TYPE":"DMT_START_COUNTS"})
    df_dmt_start_mild = df_dmt_start.loc[df_dmt_start.DMT_START=="mild"][["PATIENT_ID","Binned_time","DMT_START_COUNTS"]].rename(columns = {"DMT_START_COUNTS":"DMT_START_MILD_COUNTS"})
    df_dmt_start_moderate = df_dmt_start.loc[df_dmt_start.DMT_START=="moderate"][["PATIENT_ID","Binned_time","DMT_START_COUNTS"]].rename(columns = {"DMT_START_COUNTS":"DMT_START_MODERATE_COUNTS"})
    df_dmt_start_high = df_dmt_start.loc[df_dmt_start.DMT_START=="high"][["PATIENT_ID","Binned_time","DMT_START_COUNTS"]].rename(columns = {"DMT_START_COUNTS":"DMT_START_HIGH_COUNTS"})

    df_dmt_end = df.groupby(["PATIENT_ID","Binned_time","DMT_END"])["DMT_END_TYPE"].sum().reset_index().rename(columns = {"DMT_END_TYPE":"DMT_END_COUNTS"})
    df_dmt_end_mild = df_dmt_end.loc[df_dmt_end.DMT_END=="mild"][["PATIENT_ID","Binned_time","DMT_END_COUNTS"]].rename(columns = {"DMT_END_COUNTS":"DMT_END_MILD_COUNTS"})
    df_dmt_end_moderate = df_dmt_end.loc[df_dmt_end.DMT_END=="moderate"][["PATIENT_ID","Binned_time","DMT_END_COUNTS"]].rename(columns = {"DMT_END_COUNTS":"DMT_END_MODERATE_COUNTS"})
    df_dmt_end_high = df_dmt_end.loc[df_dmt_end.DMT_END=="high"][["PATIENT_ID","Binned_time","DMT_END_COUNTS"]].rename(columns = {"DMT_END_COUNTS":"DMT_END_HIGH_COUNTS"})

    df_m = pd.merge(df_edss, df_rel, how = "outer", on = ["PATIENT_ID","Binned_time"])

    df_m = pd.merge(df_m, df_dmt_start_mild, how = "outer", on = ["PATIENT_ID","Binned_time"])
    df_m = pd.merge(df_m, df_dmt_start_moderate, how = "outer", on = ["PATIENT_ID","Binned_time"])
    df_m = pd.merge(df_m, df_dmt_start_high, how = "outer", on = ["PATIENT_ID","Binned_time"])
    
    df_m = pd.merge(df_m, df_dmt_end_mild, how = "outer", on = ["PATIENT_ID","Binned_time"])
    df_m = pd.merge(df_m, df_dmt_end_moderate, how = "outer", on = ["PATIENT_ID","Binned_time"])
    df_m = pd.merge(df_m, df_dmt_end_high, how = "outer", on = ["PATIENT_ID","Binned_time"])
   
    df = df_m.copy()
    df.sort_values(by=["PATIENT_ID","Binned_time"], inplace = True)
    df["ACTIVE_DMT"] = df.groupby("PATIENT_ID")["ACTIVE_DMT"].transform(lambda v : v.ffill())
   

    df["UNIQUE_ID"]=df["PATIENT_ID"]#.map(d)
    #Maximum EDSS in the test window.
    
    df_sub=df.loc[(df["Binned_time"]>=start_test_binned)&(df["Binned_time"]<end_test_binned)&(~df["EDSS"].isna())].copy()
    assert(df_sub["UNIQUE_ID"].nunique()==df["UNIQUE_ID"].nunique())
    df_next_max=df_sub.groupby("UNIQUE_ID")["EDSS"].max().reset_index().rename(columns={"EDSS":"next_max_EDSS"})
    
    #Minimum EDSS in the test window.
    #df_sub=df.loc[(df["Binned_time"]>=start_test_binned)&(df["Binned_time"]<end_test_binned)].copy()
    #assert(df_sub["UNIQUE_ID"].nunique()==df["UNIQUE_ID"].nunique())
    df_next_min=df_sub.groupby("UNIQUE_ID")["EDSS"].min().reset_index().rename(columns={"EDSS":"next_min_EDSS"})[["UNIQUE_ID","next_min_EDSS"]]

    #Next EDSS in the test window.
    df_subis=df.loc[(df["Binned_time"]>=start_test_binned)&(df["Binned_time"]<end_test_binned)&(~df["EDSS"].isna())].copy()
    assert(df_subis["UNIQUE_ID"].nunique()==df["UNIQUE_ID"].nunique())
    df_subis1=df_subis.sort_values(by=["UNIQUE_ID","Time"]).copy()[["UNIQUE_ID","EDSS"]]
    df_next=df_subis1.drop_duplicates(subset=["UNIQUE_ID"],keep="first").rename(columns={"EDSS":"next_EDSS"})


    #EDSS closest to time = 5 in test window.
    df_sub["Time_2_5"]=(df_sub["Time"].dt.days-(5*365)).abs()
    df_sub_sorted = df_sub[["UNIQUE_ID","Time_2_5","EDSS","Time"]].sort_values(by=["Time_2_5","EDSS"],ascending="True")
    df_closest = df_sub_sorted.drop_duplicates(subset=["UNIQUE_ID"],keep="first").rename(columns={"EDSS":"closest_2_5_EDSS","Time":"Time_closest_2_5"})

    #Compute min EDSS in the 6 months after 5
    df_ = df.merge(df_closest, on = "UNIQUE_ID", how = "left")
    #There was an error here ! 180 was set to 10 !
    df_min_6months_after_5 = df_.loc[((df_["Time"]-df_["Time_closest_2_5"]).dt.days<=180)&((df_["Time"]-df_["Time_closest_2_5"]).dt.days>0)&(~df_.EDSS.isna())].groupby("UNIQUE_ID")["EDSS"].min().reset_index()[["UNIQUE_ID","EDSS"]].rename(columns = {"EDSS":"min_EDSS_in_next_6_months_after_5"})
    df_next_after_5_edss = df_.loc[((df_["Time"]-df_["Time_closest_2_5"]).dt.days>180)&(~df_.EDSS.isna())].sort_values(by = ["UNIQUE_ID","Time"]).drop_duplicates(subset = ["UNIQUE_ID"],keep = "first")[["UNIQUE_ID","EDSS"]].rename(columns = {"EDSS":"next_EDSS_after_5"})
    
    #Last EDSS in the observation window.
    df_before=df.loc[(df["Binned_time"]<=end_observation_binned)&(~df.EDSS.isna())].copy()
    df_ascend=df_before.sort_values(by=["UNIQUE_ID","Time"]).copy()
    pat_before_subset=df_ascend.drop_duplicates(subset=["UNIQUE_ID"],keep="last")[["UNIQUE_ID","EDSS","MSCOURSE_AT_VISIT","DATE_OF_VISIT","ACTIVE_DMT"]].copy()
    pat_before_subset.rename(columns={"EDSS":"previous_EDSS","MSCOURSE_AT_VISIT":"previous_MSCOURSE", "ACTIVE_DMT":"last_active_dmt"},inplace=True)
    assert(pat_before_subset["UNIQUE_ID"].nunique()==df_next_max["UNIQUE_ID"].nunique())

    #Max EDSS in the observation window.
    df_sub2=df.loc[(df["Binned_time"]<=end_observation_binned)&(~df.EDSS.isna())].copy()
    assert(df_sub2["UNIQUE_ID"].nunique()==df["UNIQUE_ID"].nunique())
    df_prev_max=df_sub2.groupby("UNIQUE_ID")["EDSS"].max().reset_index().rename(columns={"EDSS":"max_previous_EDSS"})
    
    df_num_visits = df_sub2.groupby("UNIQUE_ID")["EDSS"].count().reset_index().rename(columns = {"EDSS":"number_of_visits"})

    df_mean_edss = df_sub2.groupby("UNIQUE_ID")["EDSS"].mean().reset_index().rename(columns = {"EDSS":"mean_edss"})
    
    df_prevs=pat_before_subset.merge(df_prev_max,on="UNIQUE_ID")
    df_prevs = df_prevs.merge(df_num_visits, on = "UNIQUE_ID")
    df_prevs = df_prevs.merge(df_mean_edss, on = "UNIQUE_ID")

    #We put the last observed EDSS and the next max in the same dataframe.
    pat_subset = df_next_max.merge(df_prevs,on="UNIQUE_ID")[["UNIQUE_ID","next_max_EDSS","max_previous_EDSS","previous_EDSS","previous_MSCOURSE","last_active_dmt","number_of_visits","mean_edss"]]
    pat_subset = pat_subset.merge(df_next,on="UNIQUE_ID")
    pat_subset = pat_subset.merge(df_closest,on="UNIQUE_ID")
    pat_subset = pat_subset.merge(df_next_min,on="UNIQUE_ID")
    pat_subset = pat_subset.merge(df_min_6months_after_5, on = "UNIQUE_ID", how = "left")
    pat_subset = pat_subset.merge(df_next_after_5_edss, on = "UNIQUE_ID", how = "left")

    #We now create the label.
    pat_subset["label_1"]=((pat_subset["previous_EDSS"]<=5.5)&((pat_subset["next_max_EDSS"]-pat_subset["previous_EDSS"])>=1))
    pat_subset["label_2"]=(pat_subset["previous_EDSS"]>5.5)&((pat_subset["next_max_EDSS"]-pat_subset["previous_EDSS"])>=0.5)
    pat_subset["label"]=(pat_subset["label_1"] | pat_subset["label_2"])*1
    pat_subset["threshold"] = pat_subset["previous_EDSS"]
    pat_subset.loc[pat_subset.previous_EDSS<=5.5,"threshold"] = pat_subset.loc[pat_subset.previous_EDSS<=5.5,"threshold"]+1
    pat_subset.loc[pat_subset.previous_EDSS>5.5,"threshold"] = pat_subset.loc[pat_subset.previous_EDSS>5.5,"threshold"]+0.5
    pat_subset.loc[pat_subset.previous_EDSS==0,"threshold"] = 1.5


    #Other label for 1point EDSS increase between last observed and next in test window.
    pat_subset["label_bis_1"]=((pat_subset["previous_EDSS"]<=5.5)&((pat_subset["next_EDSS"]-pat_subset["previous_EDSS"])>=1))
    pat_subset["label_bis_2"]=(pat_subset["previous_EDSS"]>5.5)&((pat_subset["next_EDSS"]-pat_subset["previous_EDSS"])>=0.5)
    pat_subset["label_bis"]=(pat_subset["label_bis_1"] | pat_subset["label_bis_2"])*1

    #Other label with respect to EDSS closest to 5.
    pat_subset["label_ter_1"]=((pat_subset["previous_EDSS"]<=5.5)&((pat_subset["closest_2_5_EDSS"]-pat_subset["previous_EDSS"])>=1))
    pat_subset["label_ter_2"]=(pat_subset["previous_EDSS"]>5.5)&((pat_subset["closest_2_5_EDSS"]-pat_subset["previous_EDSS"])>=0.5)
    pat_subset["label_ter"]=(pat_subset["label_ter_1"] | pat_subset["label_ter_2"])*1
    
    pat_subset["old_confirmed_label"] = pat_subset["threshold"]<=pat_subset["next_min_EDSS"]
    
    #Creating the confirmed label
    pat_subset["trailing_EDSS"] = pat_subset[["min_EDSS_in_next_6_months_after_5","next_EDSS_after_5"]].min(axis=1)   
    pat_subset.dropna(subset = ["trailing_EDSS"],inplace = True)
    pat_subset["confirmed_label"] = 0
    pat_subset.loc[(pat_subset.closest_2_5_EDSS>=pat_subset.threshold)&(pat_subset.trailing_EDSS>=pat_subset.threshold),"confirmed_label"]=1


    #Create dummies for the MSCOURSE.
    pat_subset=pd.concat([pat_subset, pd.get_dummies(pat_subset['previous_MSCOURSE'])], axis=1)
    
    #Create dummies for the dmt.
    pat_subset.loc[pat_subset.last_active_dmt.isna(),"last_active_dmt"] = "none"
    pat_subset=pd.concat([pat_subset.drop(columns = "last_active_dmt"), pd.get_dummies(pat_subset['last_active_dmt'])], axis=1)
    
    #Merge with dataset.
    df_clean=df.merge(pat_subset,on="UNIQUE_ID", how = "left")
   

    #Remove the patients without confirmed label.
    df_clean.dropna(subset = ["confirmed_label"],inplace = True)

    #Relapse in observation period
    dict_rel_obs = df_clean.loc[df_clean.Binned_time<=end_observation_binned].groupby("UNIQUE_ID")["Relapse_count_binned"].sum().to_dict()
    df_clean["Relapses_in_observation_period"] = df_clean["UNIQUE_ID"].map(dict_rel_obs)

    df_clean.loc[df_clean.Relapses_in_observation_period.isna(),"Relapses_in_observation_period"] = 0

    #We then only select the visits <6years
    df_clean=df_clean.loc[df_clean["Binned_time"]<end_test_binned].copy()

    #Shift is the time difference (binned) between the from onset scale and the from first visit scale
    df_clean["Binned_shift"] = df_clean["Binned_time_from_onset"]-df_clean["Binned_time"]

    return(df_clean)

def prepare_dataset_from_onset(df, ratio, start_test, end_test, full_paths = True):
    import numpy as np
    
    d=dict(zip(list(df["PATIENT_ID"].unique()),np.arange(df["PATIENT_ID"].nunique())))
    df["UNIQUE_ID"]=df["PATIENT_ID"].map(d)
    
    start_test_binned = (start_test*365) // ratio
    end_test_binned   = (end_test*365) // ratio
    df_sub=df.loc[(df["Binned_time"]>=start_test_binned)&(df["Binned_time"]<end_test_binned)].copy()

    assert(df_sub["UNIQUE_ID"].nunique()==df["UNIQUE_ID"].nunique())
    #EDSS closest to time = 4 in test window.
    df_sub["Time_2_5"]=(df_sub["Binned_time"]-(5*365)//ratio).abs()
    df_sub_sorted = df_sub[["UNIQUE_ID","Time_2_5","EDSS","first_EDSS"]].sort_values(by=["Time_2_5","EDSS"],ascending = "True") #Take lowest EDSS if there is competition.
    #Binned_time_from_onset_eval is the binned time from onset at which the target EDSS in sampled.
    df_closest = df_sub_sorted.drop_duplicates(subset=["UNIQUE_ID"],keep="first").rename(columns={"EDSS":"closest_2_5_EDSS","Binned_time_from_onset":"Binned_time_from_onset_eval"})

    df_closest["label_ter_1"]=((df_closest["first_EDSS"]<=5.5)&((df_closest["closest_2_5_EDSS"]-df_closest["first_EDSS"])>=1))
    df_closest["label_ter_2"]=(df_closest["first_EDSS"]>5.5)&((df_closest["closest_2_5_EDSS"]-df_closest["first_EDSS"])>=0.5)
    df_closest["label_ter"]=(df_closest["label_ter_1"] | df_closest["label_ter_2"])*1

    "Re-Compute the FIRST_MSCOURSE VALUE"
    df.drop(columns=["FIRST_MSCOURSE"],inplace=True)
    
    df_first_visit = df.loc[df["Binned_time"]==0,["UNIQUE_ID","MSCOURSE_AT_VISIT","EDSS","Time"]].sort_values(by="Time").drop_duplicates(subset=["UNIQUE_ID"],keep="first").rename(columns={"MSCOURSE_AT_VISIT":"FIRST_MSCOURSE","EDSS":"first_EDSS"})[["UNIQUE_ID","FIRST_MSCOURSE","first_EDSS"]]

    df_closest = df_closest.merge(df_first_visit,on="UNIQUE_ID").copy()

    df_closest = pd.concat([df_closest, pd.get_dummies(df_closest['FIRST_MSCOURSE'])],axis=1)
    df = df.merge(df_closest,on="UNIQUE_ID").copy()

    #Shift is the time difference (binned) between the from onset scale and the from first visit scale
    df["Binned_shift"] = df["Binned_time_from_onset"]-df["Binned_time"]

    if full_paths == False:
        #We then only select the visits <6years after first visit.
        df=df.loc[df["Binned_time"]<end_test_binned].copy()
        #else return the full trajectories.
    else:
        df = df.copy()
    return(df)
   

def divide_and_conquer(df_clean,df_mri):
    visits_data = df_clean[["UNIQUE_ID","Binned_time","EDSS"]].copy()
    visits_data["LABEL"] = "EDSS"
    visits_data.rename(columns = {"EDSS":"VALUE"},inplace=True)

    tens_data = visits_data.append(df_mri)

    label_map = {"EDSS":0,"T1":1,"T2":2}

    tens_data["LABEL"] = tens_data["LABEL"].map(label_map)    
    
    gender_dict = {"F":0,"M":1}
    df_clean["gender_class"]=df_clean["gender"].map(gender_dict)

    cov_data = df_clean[["UNIQUE_ID","gender_class","CIS","PP","PR","RR","SP",
                        "DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed",
                        "first_EDSS","last_EDSS","max_previous_EDSS",
                        "diff_EDSS"]].copy()
    cov_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)
    label_data = df_clean[["UNIQUE_ID","label","label_bis","label_ter","next_max_EDSS","next_EDSS","closest_2_5_EDSS"]].sort_values(by="UNIQUE_ID").copy()
    label_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)

    tens_data = tens_data[["UNIQUE_ID","LABEL","Binned_time","VALUE"]]
    
    #Normalize the data.
    means = np.zeros(tens_data["LABEL"].nunique())
    stds  = np.zeros(tens_data["LABEL"].nunique())
    for i in range(means.shape[0]):
        means[i] = tens_data.loc[tens_data["LABEL"]==i].mean()["VALUE"]
        stds[i]  = tens_data.loc[tens_data["LABEL"]==i].std()["VALUE"]
        tens_data.loc[tens_data["LABEL"]==i,"VALUE"] -= means[i]
        tens_data.loc[tens_data["LABEL"]==i,"VALUE"] /= stds[i]

    return(tens_data, cov_data, label_data) 

def divide_data(df_clean, from_onset = False):

    if from_onset:
        mat_data = df_clean[["UNIQUE_ID","EDSS","Binned_time_from_onset","Binned_shift"]].copy()
    else:
        mat_data = df_clean[["UNIQUE_ID","EDSS","Binned_time"]].copy()

    extended_mat_data = df_clean[["UNIQUE_ID","Binned_time","EDSS","Relapse_count_binned","DMT_START_MILD_COUNTS","DMT_START_MODERATE_COUNTS","DMT_START_HIGH_COUNTS","DMT_END_MILD_COUNTS","DMT_END_MODERATE_COUNTS","DMT_END_HIGH_COUNTS"]].copy()

    #Normalize the data.
    means = mat_data["EDSS"].mean()
    stds  = mat_data["EDSS"].std()
    mat_data["EDSS"] -= means
    mat_data["EDSS"] /= stds

    #Normalize the data.
    means = extended_mat_data["EDSS"].mean()
    stds  = extended_mat_data["EDSS"].std()
    extended_mat_data["EDSS"] -= means
    extended_mat_data["EDSS"] /= stds


    gender_dict = {"F":0,"M":1}
    df_pre_cov = df_clean.loc[~df_clean.EDSS.isna()].copy()
    df_pre_cov["gender_class"]=df_pre_cov["gender"].map(gender_dict)

    cov_data = df_pre_cov[["UNIQUE_ID","gender_class","CIS","PP","PR","RR","SP",
                                 "DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed",
                                  "first_EDSS","last_EDSS","max_previous_EDSS","mean_edss",
                            "diff_EDSS","Relapses_in_observation_period",
                         "number_of_visits","mild","moderate","high","none"]].copy()

    cov_data.rename(columns = {"mild":"last_dmt_mild","moderate":"last_dmt_moderate","high":"last_dmt_high","none":"last_dmt_none"}, inplace = True)
    
    cov_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)

    #remove CIS Patients--------
    cis_patients = cov_data.loc[cov_data.CIS==1,"UNIQUE_ID"].copy()

    cov_data = cov_data.loc[~cov_data.UNIQUE_ID.isin(cis_patients)].copy()
    cov_data.drop(columns = ["CIS"],inplace = True)
    mat_data = mat_data.loc[~mat_data.UNIQUE_ID.isin(cis_patients)].copy()
    extended_mat_data = extended_mat_data.loc[~extended_mat_data.UNIQUE_ID.isin(cis_patients)].copy()
    df_clean = df_clean.loc[~df_clean.UNIQUE_ID.isin(cis_patients)].copy()
    #---End remove CIS patients------
    
    label_data = df_clean[["UNIQUE_ID","label","label_bis","label_ter","next_max_EDSS","next_EDSS","closest_2_5_EDSS","confirmed_label"]].sort_values(by="UNIQUE_ID").copy()
    label_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)	
    
    return(mat_data, extended_mat_data, cov_data,label_data, df_clean)

def divide_data_Setup2(df_clean):
    mat_data = df_clean[["UNIQUE_ID","EDSS","Binned_time_from_onset","Binned_shift"]].copy()
    gender_dict = {"F":0,"M":1}
    df_clean["gender_class"]=df_clean["gender"].map(gender_dict)
    cov_data = df_clean[["UNIQUE_ID","gender_class","CIS","PP","PR","RR","SP",
                                 "DURATION_OF_MS_AT_VISIT","age_at_onset_days_computed",
                                  "first_EDSS"]].copy()
    cov_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)
    label_data = df_clean[["UNIQUE_ID","label_ter","closest_2_5_EDSS"]].sort_values(by="UNIQUE_ID").copy()
    label_data.drop_duplicates(subset=["UNIQUE_ID"],keep="first",inplace=True)	
    return(mat_data,cov_data,label_data)


if __name__=="__main__":
    print("Start")

    process_patients()
    df = process_visits(ratio=30) #Uncomment this if you want to recompute !
    df.Time = pd.to_timedelta(df.Time)
    df.Time_first2last = pd.to_timedelta(df.Time_first2last)
    
    df_sub,end_observation_binned,start_test_binned,end_test_binned = subset_patients(df, ratio=30)
    print("Preparing the dataset now ...")
     
    df_clean = prepare_dataset(df_sub, end_observation_binned,start_test_binned, end_test_binned,ratio=30)

    df_clean.to_csv("~/Data/MS/Cleaned_MSBASE/df_clean_full.csv",index=False)
    
    mat_data, extended_mat_data, cov_data, label_data, df_clean_final = divide_data(df_clean)
    mat_data.to_csv("~/Data/MS/Cleaned_MSBASE/mat_data.csv",index=False)
    extended_mat_data.to_csv("~/Data/MS/Cleaned_MSBASE/extended_mat_data.csv",index=False)
    cov_data.to_csv("~/Data/MS/Cleaned_MSBASE/cov_data.csv",index=False)
    label_data.to_csv("~/Data/MS/Cleaned_MSBASE/label_data.csv",index=False)
    df_clean_final.to_csv("~/Data/MS/Cleaned_MSBASE/df_clean_final.csv", index = False)

    mat_data.to_csv("~/Data/MS/Cleaned_MSBASE/mat_data.csv",index=False)
    cov_data.to_csv("~/Data/MS/Cleaned_MSBASE/cov_data.csv",index=False)
    label_data.to_csv("~/Data/MS/Cleaned_MSBASE/label_data.csv",index=False)
    
    tens_data.to_csv("~/Data/MS/Cleaned_MSBASE/tens_data.csv",index = False)





  

