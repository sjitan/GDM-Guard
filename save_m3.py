import json,os
import numpy as np,pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from joblib import dump
Path("models").mkdir(exist_ok=True,parents=True)
df=pd.read_csv("data/GDMGuard_dataset_v2_2.csv")
t12=df[df.stage_code.isin(["T1","T2"])].copy()
first=t12.sort_values(["subj","stage_code","gestational_week_at_capture"]).groupby("subj").head(1)
y=first["GDM_dx"].astype(int).values
base=["age","BMI","parity","prior_GDM","family_DM","chronic_HTN","prior_macrosomia","ethnicity"]
gwg=["gwg_dev_from_IOM"]
selfie=["rPPG_HR","rPPG_stability","sleep_hours_24h","neck_circ_norm"]
prs=["PRS_0to10","PRS_missing"]
X=first[base+gwg+selfie+prs].copy()
cat=["ethnicity"]
num=[c for c in X.columns if c not in cat]
ct=ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="UNK")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
],remainder="drop")
clf=Pipeline([("prep",ct),("clf",LogisticRegression(max_iter=2000,class_weight="balanced",solver="lbfgs"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
clf.fit(Xtr,ytr)
dump(clf,"models/m3.joblib")
print("models/m3.joblib")
