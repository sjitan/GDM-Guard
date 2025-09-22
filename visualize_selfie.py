import argparse,cv2,mediapipe as mp,time,collections,numpy as np,json,os
ap=argparse.ArgumentParser()
ap.add_argument("--video",required=True)
ap.add_argument("--out_json",default="sessions/last_features.json")
ap.add_argument("--duration",type=float,default=0)
args=ap.parse_args()
os.makedirs("sessions",exist_ok=True)
cap=cv2.VideoCapture(args.video)
if not cap.isOpened(): raise SystemExit("video_open_failed")
fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
du=mp.solutions.drawing_utils; ds=mp.solutions.drawing_styles
def ear(land,w,h):
    def L(i): p=land[i]; return np.array([p.x*w,p.y*h])
    U=(np.linalg.norm(L(386)-L(374))+np.linalg.norm(L(159)-L(145)))/2.0
    V=np.linalg.norm(L(263)-L(362))
    return U/(V+1e-6)
buf=collections.deque(maxlen=150)
blinks=0; prev_open=True
t0=time.time()
while True:
    ok,fr=cap.read()
    if not ok: break
    rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
    r=fm.process(rgb)
    h,w=fr.shape[:2]
    if r.multi_face_landmarks:
        lms=r.multi_face_landmarks[0]
        du.draw_landmarks(fr,lms,mp.solutions.face_mesh.FACEMESH_TESSELATION,ds.get_default_face_mesh_tesselation_style(),ds.get_default_face_mesh_tesselation_style())
        e=ear(lms.landmark,w,h)
        openv=1.0 if e>0.25 else 0.0
        buf.append(1.0-openv)
        now_open=openv>0.5
        if prev_open and not now_open: blinks+=1
        prev_open=now_open
    perclos=float(np.mean(buf)) if len(buf)>0 else 0.0
    fps_text=f"PERCLOS {perclos:.2f}  Blinks {blinks}"
    cv2.rectangle(fr,(10,10),(10+530,50),(255,255,255),-1)
    cv2.putText(fr,fps_text,(18,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
    cv2.imshow("GDMGuard Face Visualizer",fr)
    if cv2.waitKey(1)&0xFF==ord('q'): break
    if args.duration>0 and (time.time()-t0)>=args.duration: break
cap.release(); cv2.destroyAllWindows()
out={"perclos":float(np.mean(buf)) if len(buf)>0 else 0.0,"blink_rate":blinks/max(1e-3,(time.time()-t0))*60.0}
with open(args.out_json,"w") as f: json.dump(out,f)
print(args.out_json)
