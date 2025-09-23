import os, json, math, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _fv(d,k):
    v=d.get(k)
    try: return float(v) if v is not None else float("nan")
    except: return float("nan")
def _nan(x): return (x is None) or (isinstance(x,float) and math.isnan(x))

def build(intake_path="data/intake_demo.json", selfie_path="sessions/vis_metrics.json"):
    os.makedirs("sessions",exist_ok=True); os.makedirs("reports",exist_ok=True)
    intake=json.load(open(intake_path))
    vis=json.load(open(selfie_path))
    vm=vis.get("metrics",{})

    M0=intake["M0"]
    M1=intake["M1"]
    M3=intake["M3"]
    M2={"rppg_bpm_mean":_fv(vm,"rppg_bpm_mean"),"rppg_snr_mean":_fv(vm,"rppg_snr_mean"),
        "perclos":_fv(vm,"perclos"),"blink_per_min":_fv(vm,"blink_per_min"),"neck_norm":_fv(vm,"neck_norm")}

    X_row={ "age":_fv(M0,"age"),"bmi":_fv(M0,"bmi"),"parity":_fv(M0,"parity"),"prior_gdm":_fv(M0,"prior_gdm"),
            "family_dm":_fv(M0,"family_dm"),"ethnicity_bin":_fv(M0,"ethnicity_bin"),
            "gwg_slope_kg_per_wk_T1":_fv(M1,"gwg_slope_kg_per_wk_T1"),"gwg_slope_kg_per_wk_T2":_fv(M1,"gwg_slope_kg_per_wk_T2"),
            "gwg_slope_kg_per_wk_T3":_fv(M1,"gwg_slope_kg_per_wk_T3"),"gwg_dev_from_IOM":_fv(M1,"gwg_dev_from_IOM"),
            "rppg_bpm_mean":_fv(M2,"rppg_bpm_mean"),"rppg_snr_mean":_fv(M2,"rppg_snr_mean"),
            "perclos":_fv(M2,"perclos"),"blink_per_min":_fv(M2,"blink_per_min"),"neck_norm":_fv(M2,"neck_norm"),
            "PRS_0to10":_fv(M3,"PRS_0to10"),"PRS_missing":_fv(M3,"PRS_missing") }

    if not os.path.exists("data/fake_cohort_T1_T4.csv"):
        import numpy as np
        rng=np.random.default_rng(42); n=4000
        age=rng.normal(30,5,n).clip(18,45); bmi=rng.normal(27,5,n).clip(18,45)
        parity=rng.integers(0,3,n); prior_gdm=rng.integers(0,2,n); family_dm=rng.integers(0,2,n); ethnicity_bin=rng.integers(0,2,n)
        gwg1=rng.normal(0.35,0.15,n).clip(-0.2,0.9); gwg2=rng.normal(0.45,0.18,n).clip(-0.2,1.1)
        gwg3=rng.normal(0.30,0.15,n).clip(-0.3,0.9); gwg_dev=rng.normal(0.8,0.8,n).clip(-2.0,3.0)
        bpm=rng.normal(82,9,n).clip(50,140); snr=rng.normal(2.1,0.7,n).clip(0,6)
        per=rng.normal(0.22,0.08,n).clip(0,1); bln=rng.normal(14,5,n).clip(0,45); neck=rng.normal(2.0,0.25,n).clip(1.2,3.2)
        prs=rng.integers(0,11,n); prs_missing=(rng.random(n)<0.1).astype(int)
        def z(x): return (x-x.mean())/(x.std()+1e-6)
        lin_pp= -2.0 + 0.05*z(age) + 0.10*z(bmi) + 0.30*prior_gdm + 0.18*family_dm + 0.12*z(gwg2) + 0.10*z(gwg_dev) + 0.08*z(bpm) + 0.14*z(per) + 0.08*z(neck) + 0.06*(prs/10) - 0.10*z(snr)
        pp=(rng.random(n)<(1/(1+np.exp(-lin_pp)))).astype(int)
        lin_t2d= -3.0 + 0.05*z(bmi) + 0.20*prior_gdm + 0.10*family_dm + 0.10*z(gwg3) + 0.15*z(per) + 0.10*z(neck) + 0.12*(prs/10) - 0.08*z(snr)
        t2d=(rng.random(n)<(1/(1+np.exp(-lin_t2d)))).astype(int)
        pd.DataFrame(dict(
            age=age,bmi=bmi,parity=parity,prior_gdm=prior_gdm,family_dm=family_dm,ethnicity_bin=ethnicity_bin,
            gwg_slope_kg_per_wk_T1=gwg1,gwg_slope_kg_per_wk_T2=gwg2,gwg_slope_kg_per_wk_T3=gwg3,gwg_dev_from_IOM=gwg_dev,
            rppg_bpm_mean=bpm,rppg_snr_mean=snr,perclos=per,blink_per_min=bln,neck_norm=neck,
            PRS_0to10=prs,PRS_missing=prs_missing,
            pp_glucose_test_done_T4=pp,incident_T2D_12m=t2d
        )).to_csv("data/fake_cohort_T1_T4.csv",index=False)

    base=pd.read_csv("data/fake_cohort_T1_T4.csv")
    feats=list(X_row.keys())
    y1="pp_glucose_test_done_T4"; y2="incident_T2D_12m"
    X=base[feats].copy(); y_pp=base[y1].astype(int); y_t2d=base[y2].astype(int)
    ct=ColumnTransformer([("num",StandardScaler(),feats)],remainder="drop")
    pp=Pipeline([("ct",ct),("lr",LogisticRegression(max_iter=400))]); t2d=Pipeline([("ct",ct),("lr",LogisticRegression(max_iter=400))])
    pp.fit(X,y_pp); t2d.fit(X,y_t2d)
    xq=pd.DataFrame([X_row])
    pp_prob=float(pp.predict_proba(xq)[0,1]); t2d_prob=float(t2d.predict_proba(xq)[0,1])
    risk="high" if (t2d_prob>=0.35 or pp_prob<0.55) else ("moderate" if (t2d_prob>=0.20 or pp_prob<0.70) else "low")
    rec="Elevated risk; schedule follow-up and ensure T4 lab order now." if risk=="high" else ("Borderline; recapture selfie and reinforce postpartum testing reminder." if risk=="moderate" else "Within baseline; routine follow-through.")
    stack={"M0":M0,"M1":M1,"M2":M2,"M3":M3}
    report={"subject_id":intake.get("subject_id","NA"),"risk_level":risk,"pp_glucose_test_done_T4_prob":pp_prob,"incident_T2D_12m_prob":t2d_prob,"M_layers":stack,"recommendation":rec}
    json.dump(stack,open("sessions/M_stack.json","w"),indent=2)
    json.dump(report,open("sessions/recommendation.json","w"),indent=2)
    open("sessions/recommendation_report.txt","w").write(
        f"Subject: {report['subject_id']}\nOutcome: {risk} — {rec}\nT4 test prob: {pp_prob:.2f}\n12m T2D prob: {t2d_prob:.2f}\n"
    )
    fig,axes=plt.subplots(1,4,figsize=(19,4.8),constrained_layout=True)
    def panel(ax, series, current, title):
        s=series.dropna(); ax.hist(s,bins=40,alpha=0.75)
        if not _nan(current): ax.axvline(current,linewidth=3)
        mu=s.mean(); sd=s.std(ddof=1) or 1.0; z=(current-mu)/sd if not _nan(current) else np.nan
        ax.set_title(f"{title} (z={0 if np.isnan(z) else round(z,2)})"); ax.grid(True,alpha=0.2)
    panel(axes[0], base["rppg_bpm_mean"], X_row["rppg_bpm_mean"], "BPM")
    panel(axes[1], base["perclos"],        X_row["perclos"],        "PERCLOS")
    panel(axes[2], base["neck_norm"],      X_row["neck_norm"],      "Neck")
    panel(axes[3], base["rppg_snr_mean"],  X_row["rppg_snr_mean"],  "SNR")
    fig.suptitle("GDM-Guard — M2 selfie vs synthetic cohort"); fig.savefig("sessions/assessment.png",dpi=150)
    ts=pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([{**{"ts":ts,"risk_level":risk,"recommendation":rec,"pp_prob":pp_prob,"t2d_prob":t2d_prob}, **X_row}]).to_csv(f"sessions/session_{ts}.csv",index=False)
    return report
