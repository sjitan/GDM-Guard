import argparse,json,time,os,numpy as np,cv2,mediapipe as mp
ap=argparse.ArgumentParser()
ap.add_argument("--video",required=True)
ap.add_argument("--duration",type=int,default=12)
ap.add_argument("--mirror",type=int,default=1)
ap.add_argument("--out",type=str,default="features_selfie.json")
args=ap.parse_args()
fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
du=mp.solutions.drawing_utils; ds=mp.solutions.drawing_styles
cap=cv2.VideoCapture(args.video)
if not cap.isOpened(): raise SystemExit("video_open_failed")
def ear(land,w,h):
    def L(i): p=land[i]; return np.array([p.x*w,p.y*h])
    U=(np.linalg.norm(L(386)-L(374))+np.linalg.norm(L(159)-L(145)))/2.0
    V=np.linalg.norm(L(263)-L(362))
    return U/(V+1e-6)
def neck_proxy(land,w,h):
    idx=[152,377,400,378,379,365,397,288,435,432,422,436,152,234,167,164,393,391,365]
    xs=[land[i].x*w for i in idx]
    return float(np.clip((max(xs)-min(xs))/(w+1e-6)/0.22,0.8,1.4))
perclos=[]; blinks=0; prev_open=True; nk=1.0; t0=time.time()
while True:
    ok,fr=cap.read()
    if not ok: break
    if args.mirror: fr=cv2.flip(fr,1)
    h,w=fr.shape[:2]
    r=fm.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    if r.multi_face_landmarks:
        lms=r.multi_face_landmarks[0].landmark
        e=ear(lms,w,h)
        openv=1.0 if e>0.25 else 0.0
        perclos.append(1.0-openv)
        now_open=openv>0.5
        if prev_open and not now_open: blinks+=1
        prev_open=now_open
        nk=neck_proxy(lms,w,h)
        du.draw_landmarks(fr,r.multi_face_landmarks[0],mp.solutions.face_mesh.FACEMESH_TESSELATION,ds.get_default_face_mesh_tesselation_style(),ds.get_default_face_mesh_tesselation_style())
    else:
        perclos.append(1.0); nk=1.0; prev_open=False
    elapsed=time.time()-t0
    txt=f"t {int(elapsed)}s  PERCLOS {np.mean(perclos):.2f}  blinks/min {(blinks/max(1e-3,elapsed)*60):.1f}  neck {nk:.2f}"
    cv2.rectangle(fr,(10,10),(10+len(txt)*9,50),(0,0,0),-1)
    cv2.putText(fr,txt,(18,42),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("GDMGuard Selfie",fr)
    if cv2.waitKey(1)&0xFF==27: break
    if elapsed>=args.duration: break
cap.release(); cv2.destroyAllWindows()
p=float(np.clip(np.mean(perclos) if perclos else 0.2,0,1))
dur=max(1e-3,time.time()-t0); br=blinks/dur*60.0; nk=float(nk)
os.makedirs("sessions",exist_ok=True)
with open(args.out,"w") as f: json.dump({"perclos":p,"blink_rate":br,"neck_norm":nk},f)
print({"perclos":p,"blink_rate":br,"neck_norm":nk})
