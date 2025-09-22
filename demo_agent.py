import pandas as pd,matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import os
p=None
for c in ["reports/risk_M3.csv","reports/risk_M2.csv","reports/risk_M1.csv","reports/risk_M0.csv"]:
    if os.path.exists(c): p=c; break
if p is None: raise SystemExit("no risk csv")
df=pd.read_csv(p).sort_values("prob",ascending=False)
top=df.head(10)
plt.figure(figsize=(6,4)); plt.barh([str(int(x)) for x in top["subj"]][::-1], top["prob"][::-1]); plt.xlabel("Predicted risk"); plt.ylabel("Subject"); plt.tight_layout(); plt.savefig("reports/demo_top10.png"); plt.close()
top.iloc[:1].to_csv("reports/demo_next_steps.txt",index=False)
print("reports/demo_top10.png reports/demo_next_steps.txt")
