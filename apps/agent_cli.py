import argparse, json, time, math
def load_feats(p): 
    try: return json.load(open(p))
    except: return {"perclos":None,"blink_rate":None,"neck_norm":None}
def clamp01(x): return max(0.0,min(1.0,x))
ap=argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--duration", type=float, default=12)
ap.add_argument("--mirror", type=int, default=1)
ap.add_argument("--age", type=float, required=True)
ap.add_argument("--bmi", type=float, required=True)
ap.add_argument("--parity", type=int, required=True)
ap.add_argument("--ethnicity", required=True, choices=["White","Black","Hispanic","Asian","Other"])
ap.add_argument("--prior_gdm", type=int, required=True)
ap.add_argument("--family_dm", type=int, required=True)
ap.add_argument("--feat_path", default="sessions/run_features.json")
ap.add_argument("--out", default="sessions/session_last.json")
args=ap.parse_args()
f=load_feats(args.feat_path)
BMI=args.bmi
FG=85.0
A1C=5.2
GWG=0.0
PRS=5.0
NECK=f.get("neck_norm") if f.get("neck_norm") is not None else 1.0
EDE=0.0
SLP=7.0
STAB=0.5
AGE=args.age
ETH_SHIFT={"White":0.0,"Asian":0.05,"Hispanic":0.10,"Black":0.12,"Other":0.02}[args.ethnicity]
logit=-3.2
logit+=0.07*(BMI-25)
logit+=0.04*(FG-85)
logit+=0.35*((A1C-5.2)/0.3)
logit+=0.80*(1 if args.prior_gdm==1 else 0)
logit+=0.30*(1 if args.family_dm==1 else 0)
logit+=0.50*GWG
logit+=0.15*((PRS-5)/2.0)
logit+=0.18*((NECK-1.0)/0.1)
logit+=0.10*(EDE/0.2)
logit+=0.20*((7.0-SLP)/2.0)
logit+=0.12*((0.5-STAB)/0.25)
logit+=0.10*((AGE-30)/5.0)
logit+=ETH_SHIFT
p=1/(1+math.exp(-logit))
p=clamp01(p)
tier="LOW"
if p>=0.12: tier="MED"
if p>=0.18: tier="HIGH"
bumps=0
if NECK is not None and NECK>=1.30: bumps+=1
if args.prior_gdm==1 or args.family_dm==1: bumps+=1
tiers=["LOW","MED","HIGH"]
tier=tiers[min(2,tiers.index(tier)+bumps)]
steps=[]
if tier=="HIGH":
    steps=["Early OGTT 16–20w","Nutrition consult in 7d","Auto-enroll postpartum 6–12w glucose test"]
elif tier=="MED":
    steps=["OGTT at 24–28w (consider earlier if feasible)","Lifestyle packet","Auto-enroll postpartum 6–12w glucose test"]
else:
    steps=["Lifestyle packet","Auto-enroll postpartum 6–12w glucose test"]
out={"intake":{"age":AGE,"bmi":BMI,"parity":args.parity,"ethnicity":args.ethnicity,"prior_gdm":args.prior_gdm,"family_dm":args.family_dm},
     "rec":{"risk_prob":round(p,3),"risk_tier":tier,"features":{"perclos":f.get("perclos"),"blink_rate":f.get("blink_rate"),"neck_norm":NECK},"next_steps":steps}}
open(args.out,"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
