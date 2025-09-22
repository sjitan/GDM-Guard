import argparse,json,os,subprocess,sys
ap=argparse.ArgumentParser()
ap.add_argument("--video",required=True)
ap.add_argument("--duration",type=int,default=12)
ap.add_argument("--mirror",type=int,default=1)
ap.add_argument("--age",type=float,required=True)
ap.add_argument("--bmi",type=float,required=True)
ap.add_argument("--parity",type=int,required=True)
ap.add_argument("--ethnicity",type=str,required=True)
ap.add_argument("--prior_gdm",type=int,choices=[0,1],required=True)
ap.add_argument("--family_dm",type=int,choices=[0,1],required=True)
args=ap.parse_args()
outf="sessions/run_features.json"
cmd=[sys.executable,"extract_selfie_features.py","--video",args.video,"--duration",str(args.duration),"--mirror",str(args.mirror),"--out",outf]
subprocess.check_call(cmd)
feat=json.load(open(outf))
age=args.age; bmi=args.bmi; par=args.parity; eth=args.ethnicity
prior=args.prior_gdm; fam=args.family_dm
perclos=feat.get("perclos",0.3); br=feat.get("blink_rate",18.0); neck=feat.get("neck_norm",1.0)
eth_shift={"White":0.0,"Asian":0.05,"Hispanic":0.10,"Black":0.12}.get(eth,0.02)
logit=-3.2
logit+=0.07*(bmi-25)
logit+=0.30*prior
logit+=0.30*fam
logit+=0.12*((neck-1.0)/0.1)
logit+=0.20*((perclos-0.2)/0.2)
logit+=0.05*((br-20)/10.0)
logit+=0.10*((age-30)/5.0)
logit+=eth_shift
import math
p=1/(1+math.exp(-logit))
tier=("LOW" if p<0.15 else "MED" if p<0.30 else "HIGH")
next_steps=[]
if tier!="LOW": next_steps.append("Order early OGTT 16–20w")
next_steps.append("Nutrition consult in 7 days" if tier!="LOW" else "Lifestyle packet")
next_steps.append("Auto-enroll postpartum 6–12w glucose test")
rec={"risk_prob":round(p,3),"risk_tier":tier,"features":feat,"next_steps":next_steps}
os.makedirs("sessions",exist_ok=True)
with open("sessions/session_last.json","w") as f: json.dump({"intake":{"age":age,"bmi":bmi,"parity":par,"ethnicity":eth,"prior_gdm":prior,"family_dm":fam},"rec":rec},f,indent=2)
print(json.dumps(rec,indent=2))
