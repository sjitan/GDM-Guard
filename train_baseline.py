import os,json
import numpy as np,pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve,precision_recall_curve
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
Path("reports").mkdir(parents=True,exist_ok=True)
df=pd.read_csv("data/GDMGuard_dataset_v2_2.csv")
t12=df[df.stage_code.isin(["T1","T2"])].copy()
first=t12.sort_values(["subj","stage_code","gestational_week_at_capture"]).groupby("subj").head(1)
y=first["GDM_dx"].astype(int).values
base_cols=["age","BMI","parity","prior_GDM","family_DM","chronic_HTN","prior_macrosomia","ethnicity"]
gwg_cols=["gwg_dev_from_IOM"]
selfie_cols=["rPPG_HR","rPPG_stability","sleep_hours_24h","neck_circ_norm"]
prs_cols=["PRS_0to10","PRS_missing"]
X0=first[base_cols].copy()
X1=first[base_cols+gwg_cols].copy()
X2=first[base_cols+gwg_cols+selfie_cols].copy()
X3=first[base_cols+gwg_cols+selfie_cols+prs_cols].copy()
def train_eval(name,X):
    cat=["ethnicity"]; num=[c for c in X.columns if c not in cat]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    ct=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="UNK")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ],remainder="drop")
    clf=Pipeline([("prep",ct),("clf",LogisticRegression(max_iter=2000,class_weight="balanced",solver="lbfgs"))])
    clf.fit(X_train,y_train)
    p=clf.predict_proba(X_test)[:,1]
    auc=roc_auc_score(y_test,p); ap=average_precision_score(y_test,p); bs=brier_score_loss(y_test,p)
    fpr,tpr,_=roc_curve(y_test,p); prec,rec,_=precision_recall_curve(y_test,p)
    plt.figure(figsize=(5,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--"); plt.title(f"ROC {name} AUC={auc:.3f}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.savefig(f"reports/roc_{name}.png"); plt.close()
    plt.figure(figsize=(5,4)); plt.plot(rec,prec); plt.title(f"PR {name} AP={ap:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout(); plt.savefig(f"reports/pr_{name}.png"); plt.close()
    bins=np.linspace(0,1,11); dfc=pd.DataFrame({"y":y_test,"p":p}); dfc["bin"]=pd.cut(dfc["p"],bins,include_lowest=True,ordered=True)
    cal=dfc.groupby("bin",observed=False).agg(obs=("y","mean"),pred=("p","mean")).dropna()
    plt.figure(figsize=(5,4)); plt.plot([0,1],[0,1],"--"); plt.plot(cal["pred"],cal["obs"],marker="o"); plt.title(f"Calibration {name} Brier={bs:.3f}"); plt.xlabel("Mean predicted"); plt.ylabel("Observed"); plt.tight_layout(); plt.savefig(f"reports/cal_{name}.png"); plt.close()
    p_all=clf.predict_proba(X)[:,1]
    df_all=first[["subj"]].copy(); df_all["prob"]=p_all
    cuts=[0.10,0.20,0.30]
    def tier(v):
        if v>=cuts[2]: return "High"
        if v>=cuts[1]: return "Moderate"
        if v>=cuts[0]: return "Low"
        return "Very Low"
    df_all["risk_tier"]=df_all["prob"].apply(tier)
    step_map={"High":"Early OGTT 16–20w; nutrition consult now; postpartum test auto-enrolled","Moderate":"Discuss early OGTT; nutrition referral; enroll for postpartum test","Low":"Standard 24–28w OGTT; routine counseling; postpartum test reminder","Very Low":"Standard care; postpartum test reminder"}
    df_all["next_steps"]=df_all["risk_tier"].map(step_map)
    out_csv=f"reports/risk_{name}.csv"; df_all.to_csv(out_csv,index=False)
    return {"auc":auc,"ap":ap,"brier":bs,"risk_csv":out_csv}
res={}; res["M0"]=train_eval("M0",X0); res["M1"]=train_eval("M1",X1); res["M2"]=train_eval("M2",X2); res["M3"]=train_eval("M3",X3)
with open("reports/model_metrics.json","w") as f: json.dump(res,f,indent=2)
