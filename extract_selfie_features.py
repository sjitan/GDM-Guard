import argparse,os,sys,time,math,json
import numpy as np,cv2,mediapipe as mp
from scipy.signal import butter,filtfilt,welch,detrend
def bandpass(x,fs,lo,hi):
    b,a=butter(3,[lo/(fs/2.0),hi/(fs/2.0)],btype="bandpass")
    return filtfilt(b,a,detrend(x))
def rppg(ts,fs):
    if len(ts)<30: return np.nan,0.0
    x=bandpass(np.asarray(ts,dtype=np.float32)-np.mean(ts),fs,0.7,3.0)
    f,P=welch(x,fs=fs,nperseg=min(256,len(x)))
    m=(f>=0.7)&(f<=3.0)
    if not np.any(m): return np.nan,0.0
    fi=f[m]; Pi=P[m]; k=int(np.argmax(Pi)); pk=Pi[k]; fb=fi[k]
    side=np.mean(Pi[max(0,k-3):k]+Pi[k+1:min(len(Pi),k+4)]) if len(Pi)>7 else np.mean(Pi)
    snr=10*np.log10((pk+1e-9)/(side+1e-9))
    return float(fb*60.0),float(max(0.0,min(20.0,snr))/10.0)
def eye_open_ratio(l,w,h):
    def p(i):q=l[i];return np.array([q.x*w,q.y*h])
    L=(np.linalg.norm(p(159)-p(145))+np.linalg.norm(p(386)-p(374)))/2.0
    D=np.linalg.norm(p(33)-p(263))
    return float(L/(D+1e-6))
def neck_ratio(l,w,h):
    def p(i):q=l[i];return np.array([q.x*w,q.y*h])
    jaw=np.linalg.norm(p(234)-p(454))
    pup=np.linalg.norm(p(468)-p(473))+1e-6
    return float(np.clip(jaw/pup/6.0,0.8,1.3))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",type=str,default="")
    ap.add_argument("--device",type=int,default=-1)
    ap.add_argument('--duration',type=int,default=20)
ap.add_argument('--mirror',type=int,default=1)
    ap.add_argument("--out",type=str,default="features_selfie.json")
    args=ap.parse_args()
    if args.video:
        cap=cv2.VideoCapture(args.video)
    else:
        dev=0 if args.device<0 else args.device
        cap=cv2.VideoCapture(dev)
    if not cap or not cap.isOpened(): print("open_failed"); sys.exit(1)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<=0: fps=30.0
    fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
    t0=time.time()
    g_face=[]; g_center=[]; necks=[]; opens=[]; blinks=0; prev_open=True; frames=0
    while True:
        if args.video:
            ok,fr=cap.read()
    if not ok: break
    if a.mirror: fr=cv2.flip(fr,1)
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            if t>=args.duration: break
        else:
            ok,fr=cap.read()
            if not ok: continue
            if time.time()-t0>=args.duration: break
        frames+=1
        h,w=fr.shape[:2]
        cx,cy=w//2,h//2
        cw,ch=int(w*0.25),int(h*0.25)
        roi=fr[max(0,cy-ch//2):min(h,cy+ch//2),max(0,cx-cw//2):min(w,cx+cw//2)]
        if roi.size>0: g_center.append(float(np.mean(roi[:,:,1])))
        rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
        r=fm.process(rgb)
        if r.multi_face_landmarks:
            l=r.multi_face_landmarks[0].landmark
            mask=np.zeros((h,w),dtype=np.uint8)
            idx=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
            pts=np.array([[int(l[i].x*w),int(l[i].y*h)] for i in idx if 0<=i<468],np.int32)
            if len(pts)>2:
                cv2.fillConvexPoly(mask,pts,255)
                g_face.append(float(cv2.mean(fr,mask=mask)[1]))
            eo=eye_open_ratio(l,w,h)
            now_open=eo>0.25
            if prev_open and not now_open: blinks+=1
            prev_open=now_open
            opens.append(1.0-now_open)
            try:
                necks.append(neck_ratio(l,w,h))
            except:
                pass
    cap.release()
    use=g_face if len(g_face)>=0.3*frames else g_center
    bpm,snr=rppg(use,float(fps)) if len(use)>0 else (np.nan,0.0)
    perclos=float(np.mean(opens)) if len(opens)>0 else np.nan
    blink=float(blinks/max(1e-3,args.duration)*60.0)
    neck=float(np.mean(necks)) if len(necks)>0 else np.nan
    out={"rPPG_HR":bpm,"rPPG_stability":snr,"sleep_hours_24h":np.nan,"perclos":perclos,"blink_rate_min":blink,"neck_circ_norm":neck}
    json.dump(out,open(args.out,"w"))
    print(args.out)
if __name__=="__main__":
    main()
