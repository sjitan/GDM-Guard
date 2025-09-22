import argparse, time, json, numpy as np, cv2, mediapipe as mp
from lib.rppg_utils import RPPG, forehead_roi, ear_from_mesh, perclos_from_ears, neck_proxy
import os,sys; sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
ap=argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--duration", type=float, default=20)
ap.add_argument("--out", required=True)
args=ap.parse_args()
cap=cv2.VideoCapture(args.video)
if not cap.isOpened(): raise SystemExit("video_open_failed")
fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
rppg=RPPG()
t0=time.time()
blinks=0; prev_open=True; perclos_vals=[]; neck_vals=[]; hr_vals=[]; snr_vals=[]
last_t=t0
while True:
    ok,fr=cap.read()
    if not ok: break
    t=time.time()
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    r=fm.process(rgb)
    if r.multi_face_landmarks:
        land=r.multi_face_landmarks[0].landmark
        h,w=fr.shape[:2]
        eL=ear_from_mesh(land,w,h,True); eR=ear_from_mesh(land,w,h,False)
        openv=1.0 if 0.5*(eL+eR)>0.25 else 0.0
        perclos_vals.append(1.0-openv)
        if prev_open and openv<0.5: blinks+=1
        prev_open=openv>=0.5
        npx=neck_proxy(fr,land); neck_vals.append(npx)
        roi=forehead_roi(fr,land)
        if roi is not None:
            roi_img,_=roi
            est=rppg.update(t, roi_img)
            if est is not None:
                hr,snr=est; hr_vals.append(hr); snr_vals.append(snr)
    if t-t0>=args.duration: break
cap.release()
dur=max(1e-6, time.time()-t0)
perclos=float(np.clip(np.mean(perclos_vals) if perclos_vals else 0.2,0,1))
blink_rate=float(blinks/dur*60.0)
neck_norm=float(np.median(neck_vals) if neck_vals else 1.0)
hr_bpm=float(np.median(hr_vals) if hr_vals else 0.0)
hr_snr=float(np.median(snr_vals) if snr_vals else 0.0)
out=dict(perclos=perclos, blink_rate_min=blink_rate, neck_norm=neck_norm, rppg_hr_bpm=hr_bpm, rppg_snr=hr_snr, seconds=dur)
with open(args.out,"w") as f: f.write(json.dumps(out))
print(json.dumps(out))