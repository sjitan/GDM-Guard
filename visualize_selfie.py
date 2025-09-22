import argparse, time, json, cv2, numpy as np, mediapipe as mp
ap=argparse.ArgumentParser()
ap.add_argument("--video", default="")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--duration", type=int, default=20)
ap.add_argument("--mirror", type=int, default=1)
ap.add_argument("--out_json", default="sessions/vis_metrics.json")
args=ap.parse_args()
mpf=mp.solutions.face_mesh
fm=mpf.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
du=mp.solutions.drawing_utils
ds=mp.solutions.drawing_styles
cap=cv2.VideoCapture(args.video if args.video else args.device)
if not cap.isOpened(): raise SystemExit("video_open_failed")
perclos=[]
blinks=0
prev_open=True
t0=time.time()
def ear(land,w,h):
    def L(i):
        p=land[i]; return np.array([p.x*w,p.y*h],dtype=np.float32)
    U=(np.linalg.norm(L(386)-L(374))+np.linalg.norm(L(159)-L(145)))/2.0
    V=np.linalg.norm(L(263)-L(362))
    return U/(V+1e-6)
def neck_proxy(land,w,h):
    xs=[]; ys=[]
    for i,p in enumerate(land):
        xs.append(p.x*w); ys.append(p.y*h)
    xs=np.array(xs); ys=np.array(ys)
    bw=float(xs.max()-xs.min()); bh=float(ys.max()-ys.min())
    r=bw/(bh+1e-6)
    return float(np.clip(1.0+0.8*(r-0.75),0.7,1.6))
while True:
    ok,fr=cap.read()
    if not ok: break
    if args.mirror: fr=cv2.flip(fr,1)
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    r=fm.process(rgb)
    h,w=fr.shape[:2]
    if r.multi_face_landmarks:
        lms=r.multi_face_landmarks[0].landmark
        e=ear(lms,w,h)
        openv=1.0 if e>0.25 else 0.0
        perclos.append(1.0-openv)
        now_open=openv>0.5
        if prev_open and not now_open: blinks+=1
        prev_open=now_open
        nk=neck_proxy(lms,w,h)
        du.draw_landmarks(fr,r.multi_face_landmarks[0],mpf.FACEMESH_TESSELATION,ds.get_default_face_mesh_tesselation_style(),ds.get_default_face_mesh_tesselation_style())
    else:
        perclos.append(1.0)
        nk=1.0
        prev_open=False
    elapsed=time.time()-t0
    left=max(0,int(args.duration-elapsed))
    txt=f"t {int(elapsed)}s  PERCLOS {np.mean(perclos):.2f}  blinks/min {(blinks/max(1e-3,elapsed)*60):.1f}  neck {nk:.2f}"
    cv2.rectangle(fr,(10,10),(10+int(8*len(txt)),50),(0,0,0),-1)
    cv2.putText(fr,txt,(18,42),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("GDMGuard Selfie",fr)
    if cv2.waitKey(1)&0xFF==27: break
    if elapsed>=args.duration: break
cap.release()
cv2.destroyAllWindows()
p=float(np.clip(np.mean(perclos) if perclos else 0.2,0,1))
dur=max(1e-3,time.time()-t0)
br=blinks/dur*60.0
nk=float(nk if 'nk' in locals() else 1.0)
out=dict(perclos=float(p), blink_rate=float(br), neck_norm=float(nk))
import os
os.makedirs("sessions",exist_ok=True)
with open(args.out_json,"w") as f: json.dump(out,f)
print(out)
