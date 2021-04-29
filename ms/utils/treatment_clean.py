
import pandas as pd
from ms import DATA_DIR
from joblib import Parallel, delayed
import multiprocessing

def dmt_clean_fun():
    df = pd.read_csv(DATA_DIR + "/treatment.csv")
    df.dropna(subset = ["dmt"], inplace = True)

    df["dmt_clean"] = df["dmt"]

    df.loc[df.dmt.str.contains("MS",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("TY",case = False),"dmt_clean"] = "Natalizumab"
    df.loc[df.dmt.str.contains("FTY",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("CLAD",case = False),"dmt_clean"] = "Cladribine"
    df.loc[df.dmt.str.contains("Move",case = False),"dmt_clean"] = "Cladribine"

    df.loc[df.dmt.str.contains("RITU",case = False),"dmt_clean"] = "Rituximab"
    df.loc[df.dmt.str.contains("MABT",case = False),"dmt_clean"] = "Rituximab"
    df.loc[df.dmt.str.contains("MAB-T",case = False),"dmt_clean"] = "Rituximab"
    df.loc[df.dmt.str.contains("RITO",case = False),"dmt_clean"] = "Rituximab"

    df.loc[df.dmt.str.contains("OCRELIZUMAB",case = False),"dmt_clean"] = "Ocrelizumab"
    df.loc[df.dmt.str.contains("OCREVUS",case = False),"dmt_clean"] = "Ocrelizumab"

    df.loc[df.dmt.str.contains("TYSABRI",case = False),"dmt_clean"] = "Natalizumab"
    df.loc[df.dmt.str.contains("Nata",case = False),"dmt_clean"] = "Natalizumab"

    df.loc[df.dmt.str.contains("Gla",case = False),"dmt_clean"] = "Glatiramer"
    df.loc[df.dmt.str.contains("SINOMER",case = False),"dmt_clean"] = "Glatiramer"

    df.loc[df.dmt.str.contains("LEM",case = False),"dmt_clean"] = "Alemtuzumab"
    df.loc[df.dmt.str.contains("ALENTUZUMAB",case = False),"dmt_clean"] = "Alemtuzumab"

    df.loc[df.dmt.str.contains("SIPONIMOD",case = False),"dmt_clean"] = "Siponimod"
    df.loc[df.dmt.str.contains("Teri",case = False),"dmt_clean"] = "Teriflunomide"
    df.loc[df.dmt.str.contains("AUBA",case = False),"dmt_clean"] = "Teriflunomide"
    df.loc[df.dmt.str.contains("TEROFLUNAMIDE",case = False),"dmt_clean"] = "Teriflunomide"

    df.loc[df.dmt.str.contains("gilenya",case = False),"dmt_clean"] = "Fingolimod"
    df.loc[df.dmt.str.contains("limod",case = False),"dmt_clean"] = "Fingolimod"
    df.loc[df.dmt.str.contains("NIA",case = False),"dmt_clean"] = "Fingolimod"
    df.loc[df.dmt.str.contains("VINTOR",case = False),"dmt_clean"] = "Fingolimod"

    df.loc[df.dmt.str.contains("Copax",case = False),"dmt_clean"] = "Glatiramer"

    df.loc[df.dmt.str.contains("Interfer",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("Avonex",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("Rebif",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("Plegridy",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("IFN",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("feron",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("beta",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("EXTA",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("CINOVEX",case = False),"dmt_clean"] = "Interferons"
    df.loc[df.dmt.str.contains("REBI",case = False),"dmt_clean"] = "Interferons"

    df.loc[df.dmt.str.contains("ANTRONE",case = False),"dmt_clean"] = "Novantrone"

    df.loc[df.dmt.str.contains("Dime",case = False),"dmt_clean"] = "Dimethyl-Fumarate"
    df.loc[df.dmt.str.contains("TECF",case = False),"dmt_clean"] = "Dimethyl-Fumarate"

    df.loc[df.dmt.str.contains("Cortison",case = False),"dmt_clean"] = "Cortisone"

    df.loc[df.dmt.str.contains("Steroid",case = False),"dmt_clean"] = "Steroids"


    df.loc[df.dmt.str.contains("TREX",case = False),"dmt_clean"] = "Methotrexate"
    df.loc[df.dmt.str.contains("METHOREXATE",case = False),"dmt_clean"] = "Methotrexate"


    df.loc[df.dmt.str.contains("PRED",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("SOLU",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("SOLD",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("DELTA",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("DECA",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("DEXAMETH",case = False),"dmt_clean"] = "Glucocorticoid"
    df.loc[df.dmt.str.contains("TETRACOSACTIDE",case = False),"dmt_clean"] = "Glucocorticoid"



    df.loc[df.dmt.str.contains("AZAT",case = False),"dmt_clean"] = "Azathioprine"
    df.loc[df.dmt.str.contains("IMUR",case = False),"dmt_clean"] = "Azathioprine"
    df.loc[df.dmt.str.contains("ÝMÜRAN",case = False),"dmt_clean"] = "Azathioprine"


    df.loc[df.dmt.str.contains("ENDOX",case = False),"dmt_clean"] = "Endoxan"
    df.loc[df.dmt.str.contains("CYCLOP",case = False),"dmt_clean"] = "Endoxan"
    df.loc[df.dmt.str.contains("CICLOF",case = False),"dmt_clean"] = "Endoxan"
    df.loc[df.dmt.str.contains("ENDOKSAN",case = False),"dmt_clean"] = "Endoxan"
    df.loc[df.dmt.str.contains("SIKLOFOSFAMID",case = False),"dmt_clean"] = "Endoxan"


    df.loc[df.dmt.str.contains("DACLIZUMAB",case = False),"dmt_clean"] = "Daclizumab"
    df.loc[df.dmt.str.contains("ZINBRYTA",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("Siponimod",case = False),"dmt_clean"] = "Siponimod"

    df.loc[df.dmt.str.len()<=4,"dmt_clean"] = None
    df.loc[df.dmt.str.contains("Trans",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("CBAF",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("Glob",case = False),"dmt_clean"] = None #immunoglobulin
    df.loc[df.dmt.str.contains("GAMMA",case = False),"dmt_clean"] = None #immunoglobulin
    df.loc[df.dmt.str.contains("KIOVIG",case = False),"dmt_clean"] = None #immunoglobulin
    df.loc[df.dmt.str.contains("NANOGAM",case = False),"dmt_clean"] = None #immunoglobulin

    df.loc[df.dmt.str.contains("CHLOR",case = False),"dmt_clean"] = None #hydroxychloroquine
    df.loc[df.dmt.str.contains("CELL",case = False),"dmt_clean"] = None #stem cells
    df.loc[df.dmt.str.contains("VIT",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("BIOTIN",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("#",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("Trial",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("STUD",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("ETUD",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("ESSAI",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("WA",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("MT13",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("PROT",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("Phase",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("MONT",case = False),"dmt_clean"] = None

    df.loc[df.dmt.str.contains("Placebo",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("Vitamin",case = False),"dmt_clean"] = None
    df.loc[df.dmt.str.contains("AMPYRA",case = False),"dmt_clean"] = None

    df.loc[(df.dmt_clean.str.contains("Methotrexate",case = False)) | 
           (df.dmt_clean.str.contains("Endoxan",case = False)) | 
           (df.dmt_clean.str.contains("Azathioprine",case = False)) |
           (df.dmt_clean.str.contains("Novantrone",case = False)) |
           (df.dmt_clean.str.contains("Glucocorticoid",case = False))
           ,"dmt_clean"] = None #Casted as None in the end

    df.loc[df.dmt_clean.isin(["Interferons","Teriflunomide","Glatiramer"]),"DMT_GROUP"] = "mild"
    df.loc[df.dmt_clean.isin(["Fingolimod","Dimethyl-Fumarate","Cladribine","Siponimod","Daclizumab"]),"DMT_GROUP"] = "moderate"
    df.loc[df.dmt_clean.isin(["Alemtuzumab","Rituximab","Ocrelizumab","Natalizumab"]),"DMT_GROUP"] = "high"


    print(df.groupby("DMT_GROUP")["PATIENT_ID"].nunique())

    df = df.loc[~df.DMT_GROUP.isna()]
    df = df.loc[df["START_DATE"]!=df["END_DATE"]] #drop non-lasting DMTs
    df = df.loc[df["START_DATE"]<df["END_DATE"]]
    df = df[["PATIENT_ID","START_DATE","END_DATE","DMT_GROUP"]]


    # Check if no overlap in the DMT prescriptions.
    def no_overlap(sub_df):
        if not (sub_df["END_DATE"][1:].reset_index(drop = True) > sub_df["START_DATE"][:-1].reset_index(drop = True)).all():
            return False
        else:
            return True

    assert df.groupby("PATIENT_ID").apply(no_overlap).all()




    df.to_csv(DATA_DIR + "/treatment_clean.csv",index = False)


    def dmt_history(sub_df):
        sub_df_start = sub_df[["PATIENT_ID","START_DATE","DMT_GROUP"]].rename(columns = {"START_DATE":"date"})
        sub_df_end = sub_df[["PATIENT_ID","END_DATE"]].rename(columns = {"END_DATE":"date"})
        
        sub_merge = pd.merge(sub_df_start,sub_df_end, how = "outer", on = ["PATIENT_ID","date"]).sort_values(by=["date"])

        return sub_merge

    def applyParallel(dfGrouped, func):
        retLst = Parallel(n_jobs=20)(delayed(func)(group) for name, group in dfGrouped)
        return pd.concat(retLst)

    #Computing the DMT history of the patients
    dmtstory = applyParallel(df.groupby("PATIENT_ID"),dmt_history)
    dmtstory.rename(columns = {"DMT_GROUP":'ACTIVE_DMT'}, inplace = True)
    dmtstory.loc[dmtstory.ACTIVE_DMT.isna(),"ACTIVE_DMT"] = "none"

    dmtstory.to_csv(DATA_DIR + "/treatment_history_clean.csv",index = False)
