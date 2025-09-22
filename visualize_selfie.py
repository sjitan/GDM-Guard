import argparse,cv2,mediapipe as mp,time,collections,numpy as np,json,os
def ear(l,w,h):
    def P(i): p=l[i]; return np.array([p.x*w,p.y*h])
    U=(np.linalg.norm(P(386)-P(374))+np.linalg.norm(P(159)-P(145)))/2.0
    V=np.linalg.norm(P(263)-P(362))+1e-6
    return U/V
def neck_norm_from(l,w,h):
    def P(i): p=l[i]; return np.array([p.x*w,p.y*h])
    jawL,jawR=P(234),P(454); eyeL,eyeR=P(33),P(263)
    jaw_w=np.linalg.norm(jawR-jawL); eye_w=np.linalg.norm(eyeR-eyeL)+1e-6
    return float(np.clip(jaw_w/eye_w,0.8,1.4))
def draw_hud(fr,perclos,blinks_per_min,neck_norm,fps):
    cv2.rectangle(fr,(10,10),(820,100),(255,255,255),-1)
    cv2.putText(fr,f"FPS {fps:.1f}",(18,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
    cv2.putText(fr,f"PERCLOS {perclos:.2f}  Blinks/min {blinks_per_min:.1f}  Neck {neck_norm:.2f}",(18,80),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
    gx,gy,gw,gh=18,105,300,22
    nv=max(0.0,min(1.0,(neck_norm-0.9)/0.3))
    cv2.rectangle(fr,(gx,gy),(gx+gw,gy+gh),(50,50,50),2)
    fill=int((gw-4)*nv)
    color=(0,200,0) if nv<=0.66 else (0,165,255) if nv<=0.9 else (0,0,255)
    cv2.rectangle(fr,(gx+2,gy+2),(gx+2+fill,gy+gh-2),color,-1)
def run(video,camera,duration,out_json):
    cap=cv2.VideoCapture(video) if video else cv2.VideoCapture(camera)
    if not cap.isOpened(): raise SystemExit("camera_or_video_open_failed")
    fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    du=mp.solutions.drawing_utils; ds=mp.solutions.drawing_styles
    t0=time.time(); last=time.time(); blinks=0; was_open=True; perclos_buf=collections.deque(maxlen=300); neck_last=1.0
    while True:
        ok,fr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        r=fm.process(rgb)
        h,w=fr.shape[:2]
        eye_open=0.0; neck_norm=neck_last
        if r.multi_face_landmarks:
            lms=r.multi_face_landmarks[0].landmark
            du.draw_landmarks(fr,r.multi_face_landmarks[0],mp.solutions.face_mesh.FACEMESH_TESSELATION,ds.get_default_face_mesh_tesselation_style(),ds.get_default_face_mesh_tesselation_style())
            e=ear(lms,w,h)
            eye_open=1.0 if e>0.25 else 0.0
            neck_norm=neck_norm_from(lms,w,h)
            neck_last=neck_norm
        perclos_buf.append(1.0-eye_open)
        now_open=eye_open>0.5
        if was_open and not now_open: blinks+=1
        was_open=now_open
        dt=max(1e-3,time.time()-last); fps=1.0/dt; last=time.time()
        perclos=float(np.mean(perclos_buf)) if len(perclos_buf)>0 else 0.0
        blinks_per_min=blinks/max(1e-3,(time.time()-t0))*60.0
        draw_hud(fr,perclos,blinks_per_min,neck_norm,fps)
        cv2.imshow("GDMGuard Visualizer",fr)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        if duration>0 and time.time()-t0>=duration: break
    cap.release(); cv2.destroyAllWindows()
    out={"perclos":perclos,"blink_rate":blinks_per_min,"neck_norm":neck_norm}
    if out_json:
        os.makedirs(os.path.dirname(out_json),exist_ok=True)
        with open(out_json,"w") as f: json.dump(out,f)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",default="")
    ap.add_argument("--camera",type=int,default=0)
    ap.add_argument("--duration",type=int,default=20)
    ap.add_argument("--out_json",default="sessions/vis_metrics.json")
    a=ap.parse_args()
    run(a.video if a.video else "",a.camera,a.duration,a.out_json)
