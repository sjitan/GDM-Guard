import numpy as np, cv2
class RPPG:
    def __init__(self, min_sec=8.0, max_sec=20.0):
        self.t=[]; self.v=[]; self.min_sec=min_sec; self.max_sec=max_sec
    def _bandpass_fft(self, t, x, f_lo=0.8, f_hi=3.0):
        t=np.array(t); x=np.array(x); t=t-t[0]
        if len(t)<2: return None
        dt=np.median(np.diff(t)); fs=1.0/max(1e-6,dt)
        n=len(x); x=x-np.mean(x); x=x/np.std(x)+1e-12
        X=np.fft.rfft(x)
        f=np.fft.rfftfreq(n, d=1.0/fs)
        band=(f>=f_lo)&(f<=f_hi)
        if band.sum()<3: return None
        mag=np.abs(X)[band]
        f_band=f[band]
        idx=int(np.argmax(mag))
        f_peak=f_band[idx]
        peak=mag[idx]
        noise=np.mean(np.delete(mag, idx)) if len(mag)>1 else 1e-6
        snr=float(peak/(noise+1e-6))
        bpm=float(f_peak*60.0)
        return bpm, snr
    def update(self, ts, roi_bgr):
        g=np.mean(roi_bgr[:,:,1])
        self.t.append(float(ts)); self.v.append(float(g))
        while self.t and (self.t[-1]-self.t[0])>self.max_sec:
            self.t.pop(0); self.v.pop(0)
        if (self.t[-1]-self.t[0])<self.min_sec: return None
        out=self._bandpass_fft(self.t, self.v)
        return out
def forehead_roi(frame, lms):
    h,w=frame.shape[:2]
    xs=[int(lm.x*w) for lm in lms]; ys=[int(lm.y*h) for lm in lms]
    x1=max(0, min(xs)); x2=min(w-1, max(xs)); y1=max(0, min(ys)); y2=min(h-1, max(ys))
    bw=x2-x1; bh=y2-y1
    fx1=int(x1+bw*0.30); fx2=int(x1+bw*0.70); fy1=int(y1+bh*0.15); fy2=int(y1+bh*0.35)
    fx1=max(0,fx1); fy1=max(0,fy1); fx2=min(w-1,fx2); fy2=min(h-1,fy2)
    if fx2<=fx1 or fy2<=fy1: return None
    return frame[fy1:fy2, fx1:fx2], (fx1,fy1,fx2,fy2)
def ear_from_mesh(land,w,h,left=True):
    if left:
        ids=[386,374,159,145,263,362]
    else:
        ids=[159,145,386,374,33,133]
    pts=[(land[i].x*w, land[i].y*h) for i in ids]
    U=(np.hypot(pts[0][0]-pts[1][0],pts[0][1]-pts[1][1])+np.hypot(pts[2][0]-pts[3][0],pts[2][1]-pts[3][1]))/2.0
    V=np.hypot(pts[4][0]-pts[5][0],pts[4][1]-pts[5][1])
    return U/(V+1e-6)
def perclos_from_ears(eL,eR,thr=0.25):
    oL=1.0 if eL>thr else 0.0; oR=1.0 if eR>thr else 0.0
    return 1.0-0.5*(oL+oR)
def neck_proxy(frame,lms):
    h,w=frame.shape[:2]
    xs=[int(lm.x*w) for lm in lms]; ys=[int(lm.y*h) for lm in lms]
    x1=min(xs); x2=max(xs); y1=min(ys); y2=max(ys)
    face_w=max(1,x2-x1)
    eyeL=(int(lms[33].x*w), int(lms[33].y*h))
    eyeR=(int(lms[263].x*w), int(lms[263].y*h))
    ipd=max(1,np.hypot(eyeL[0]-eyeR[0], eyeL[1]-eyeR[1]))
    return float(face_w/ipd)
