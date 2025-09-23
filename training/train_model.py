import os,joblib,pandas as pd,numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
os.makedirs("models",exist_ok=True); os.makedirs("reports",exist_ok=True)
df=pd.read_csv("data/fake_patients_M0_M3_timeseries.csv").sort_values(["patient_id","measurement_date"]).groupby("patient_id").tail(1).reset_index(drop=True)
y=((df["bmi"]>30).astype(int)|(df["prior_gdm"]==1)|(df["family_dm"]==1)|(df["prs_bin"]==2)|(df["rppg_bpm_mean"]>95)|(df["perclos"]>0.35)|(df["neck_norm"]>2.3)).astype(int)
X=df.drop(columns=["patient_id","measurement_date"])
num=[c for c in X.columns if X[c].dtype!=object]; cat=[c for c in X.columns if X[c].dtype==object]
ct=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
clf=LogisticRegression(max_iter=200,solver="lbfgs")
pipe=Pipeline([("prep",ct),("clf",clf)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
pipe.fit(Xtr,ytr)
joblib.dump({"model":pipe,"num":num,"cat":cat,"columns":list(X.columns)},"models/m2_model.pkl")
pd.DataFrame({"metric":["train_pos_rate","test_pos_rate","n_train","n_test"],
              "value":[float(np.mean(ytr)),float(np.mean(yte)),int(len(ytr)),int(len(yte))]}).to_csv("reports/train_report.csv",index=False)
print("ok")
