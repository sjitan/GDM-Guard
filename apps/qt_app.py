import sys, os, json, time, cv2, numpy as np
from collections import deque
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QScrollArea,QTextEdit,QSplitter,QSizePolicy
from PySide6.QtCore import QTimer,Qt
from PySide6.QtGui import QImage,QPixmap,QFont
import mediapipe as mp
from lib.rppg_utils import RPPG, ear_from_mesh, neck_proxy

def to_qimage_bgr(frame):
    if frame is None: return None
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    return QImage(rgb.data,w,h,3*w,QImage.Format_RGB888)

def draw_text(img,t,x,y,c=(20,255,20)): cv2.putText(img,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2,cv2.LINE_AA)

def forehead_box(frame, lms):
    h,w,_=frame.shape
    xs=[int(p.x*w) for p in lms]; ys=[int(p.y*h) for p in lms]
    x0,x1=max(0,min(xs)),min(w-1,max(xs)); y0,y1=max(0,min(ys)),min(h-1,max(ys))
    bw=max(1,x1-x0); bh=max(1,y1-y0); cx=(x0+x1)//2
    fy=int(y0+0.18*bh); rw=int(0.32*bw); rh=int(0.16*bh)
    x=max(0,cx-rw//2); y=max(0,fy-rh//2); rw=min(rw,w-x); rh=min(rh,h-y)
    if rw<8 or rh<8: return None,None
    return frame[y:y+rh, x:x+rw].copy(),(x,y,rw,rh)

class ResultsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDM-Guard — Results")
        self.img=QLabel(alignment=Qt.AlignCenter)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.banner=QLabel(alignment=Qt.AlignCenter)
        self.banner.setStyleSheet("font: 700 16px Menlo; padding:8px;")
        self.txt=QTextEdit(); self.txt.setReadOnly(True); self.txt.setFont(QFont("Menlo",11))
        sa=QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(self.img)
        lay=QVBoxLayout(); lay.addWidget(self.banner); lay.addWidget(sa); lay.addWidget(self.txt)
        w=QWidget(); w.setLayout(lay); self.setCentralWidget(w)

    def refresh(self):
        png="sessions/assessment.png"; txtp="sessions/recommendation_report.txt"; jsonp="sessions/recommendation.json"; stack="sessions/M_stack.json"
        if os.path.exists(png):
            pix=QPixmap(png).scaled(self.img.size() if self.img.size().width()>0 else Qt.QSize(1200,800), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img.setPixmap(pix)
        else:
            self.img.setText("No plot")
        banner="Outcome: N/A"
        if os.path.exists(jsonp):
            try:
                r=json.load(open(jsonp))
                risk=r.get("risk_level","?").upper()
                rec=r.get("M_layers",{}).get("M3",{}).get("recommendation","")
                banner=f"Outcome: {risk} — {rec}  |  Stack: M0→M1→M2→M3"
            except: pass
        self.banner.setText(banner)
        body=""
        if os.path.exists(txtp): body+=open(txtp).read()
        if os.path.exists(stack):
            try: body+="\n\nM_STACK\n"+json.dumps(json.load(open(stack)),indent=2)
            except: pass
        if not body and os.path.exists(jsonp): body=open(jsonp).read()
        if not body: body="No recommendation"
        self.txt.setPlainText(body)

    def showEvent(self,e):
        self.refresh(); return super().showEvent(e)

class HUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDM-Guard HUD")
        self.vlabel=QLabel("No video", alignment=Qt.AlignCenter)
        self.vlabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vlabel.setMinimumSize(800,520)
        self.tpane=QTextEdit(); self.tpane.setReadOnly(True); self.tpane.setFont(QFont("Menlo",12)); self.tpane.setMinimumWidth(360)
        split=QSplitter(Qt.Horizontal); split.addWidget(self.vlabel); split.addWidget(self.tpane); split.setSizes([900,380])
        self.bDemo=QPushButton("Demo"); self.bUpload=QPushButton("Upload"); self.bCam=QPushButton("Camera"); self.bAnalyze=QPushButton("Analyze Now")
        self.bResults=QPushButton("Show Results"); self.bStop=QPushButton("Stop"); self.bQuit=QPushButton("Quit")
        row=QHBoxLayout(); [row.addWidget(b) for b in (self.bDemo,self.bUpload,self.bCam,self.bAnalyze,self.bResults,self.bStop,self.bQuit)]
        lay=QVBoxLayout(); lay.addLayout(row); lay.addWidget(split); w=QWidget(); w.setLayout(lay); self.setCentralWidget(w)

        self.timer=QTimer(); self.timer.timeout.connect(self.on_frame)
        self.cap=None; self.mode=None; self.path=None; self.mirror=True

        self.fm=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.6,min_tracking_confidence=0.6)
        self.draw=mp.solutions.drawing_utils; self.sty=mp.solutions.drawing_styles
        self.rppg=RPPG()

        self.start_ts=None; self.frames=0
        self.bpm_hist=[]; self.bpm_ma=[]; self.snr_hist=[]; self.neck_hist=[]
        self.ear_hist=[]; self.blink_count=0; self.eye_closed=False; self.eye_close_t=None; self.last_blink_t=0.0
        self.perclos_hist=[]
        self.last_qimg=None

        self.cal={"scale":1.0,"offset":0.0,"clamp_min":55.0,"clamp_max":120.0,"baseline_seconds":12.0,"target_bpm":75.0}

        self.bDemo.clicked.connect(self.start_demo)
        self.bUpload.clicked.connect(self.start_upload)
        self.bCam.clicked.connect(self.start_camera)
        self.bAnalyze.clicked.connect(self.run_agent_and_show)
        self.bResults.clicked.connect(self.show_results)
        self.bStop.clicked.connect(self.on_stop)
        self.bQuit.clicked.connect(self.close)

    def open_cap(self,path=None):
        if self.cap and self.cap.isOpened(): self.cap.release()
        if path and os.path.exists(path): self.cap=cv2.VideoCapture(path)
        else: self.cap=cv2.VideoCapture(0)
        return self.cap.isOpened()

    def start_demo(self): self.mode="demo"; self.path="seeds/gdm_sample.mp4"; self._start()
    def start_upload(self):
        p,_=QFileDialog.getOpenFileName(self,"Choose video","","Video Files (*.mp4 *.mov *.m4v);;All Files (*)")
        if p: self.mode="upload"; self.path=p; self._start()
    def start_camera(self): self.mode="camera"; self.path=None; self._start()

    def _start(self):
        if not self.open_cap(self.path): return
        self.start_ts=time.time(); self.frames=0
        self.bpm_hist.clear(); self.bpm_ma.clear(); self.snr_hist.clear(); self.neck_hist.clear()
        self.ear_hist.clear(); self.blink_count=0; self.eye_closed=False; self.eye_close_t=None; self.last_blink_t=0.0
        self.perclos_hist.clear()
        self.tpane.setPlainText("Running…")
        self.timer.start(33)

        try:
            demo=json.load(open("data/intake_demo.json"))
        except Exception:
            demo={"M0":{},"M1":{},"M3":{}}
        def fmt_demo(d):
            M0=d.get("M0",{}); M1=d.get("M1",{}); M3=d.get("M3",{})
            return ("M0 (core clinical)\n"
                    f"  age: {M0.get('age','-')}  bmi: {M0.get('bmi','-')}  parity: {M0.get('parity','-')}\n"
                    f"  prior_gdm: {M0.get('prior_gdm','-')}  family_dm: {M0.get('family_dm','-')}  ethnicity_bin: {M0.get('ethnicity_bin','-')}\n\n"
                    "M1 (GWG T1–T3)\n"
                    f"  gwg_T1: {M1.get('gwg_slope_kg_per_wk_T1','-')}  gwg_T2: {M1.get('gwg_slope_kg_per_wk_T2','-')}  gwg_T3: {M1.get('gwg_slope_kg_per_wk_T3','-')}\n"
                    f"  gwg_dev_from_IOM: {M1.get('gwg_dev_from_IOM','-')}\n\n"
                    "M3 (PRS)\n"
                    f"  PRS_0to10: {M3.get('PRS_0to10','-')}  PRS_missing: {M3.get('PRS_missing','-')}\n")
        self.tpane.setPlainText("M0/M1/M3 (DEMO)\n"+fmt_demo(demo))


    def finalize_metrics(self):
        if not self.start_ts: return None
        dur=max(1e-3,time.time()-self.start_ts)
        perclos=float(np.nanmean(self.perclos_hist)) if self.perclos_hist else None
        out={"mode":self.mode,"duration_sec":dur,"frames":self.frames,
             "metrics":{
                 "rppg_bpm_mean":(None if not self.bpm_hist else float(np.nanmedian(self.bpm_hist))),
                 "rppg_snr_mean":(None if not self.snr_hist else float(np.nanmedian(self.snr_hist))),
                 "blink_per_min":float(self.blink_count*60.0/dur),
                 "ear_mean":(None if not self.ear_hist else float(np.nanmean(self.ear_hist))),
                 "perclos":perclos,
                 "neck_norm":(None if not self.neck_hist else float(np.nanmedian(self.neck_hist)))
             }}
        os.makedirs("sessions",exist_ok=True)
        with open("sessions/vis_metrics.json","w") as f: json.dump(out,f,indent=2)
        return out

    def run_agent_and_show(self):
        self.finalize_metrics()
        try:
            import subprocess, sys
            subprocess.run([sys.executable,"agent/run_agent.py"],check=True)
        except: pass
        self.show_results()

    def on_stop(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened(): self.cap.release()
        self.run_agent_and_show()

    def on_frame(self):
        try:
            if not (self.cap and self.cap.isOpened()): return
            ok,fr=self.cap.read()
            if not ok: self.on_stop(); return
            if self.mirror: fr=cv2.flip(fr,1)
            self.frames+=1; now=time.time()
            rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            res=self.fm.process(rgb)

            bpm=np.nan; snr=np.nan; neck=np.nan; ear=np.nan; blink_rate=np.nan; box=None

            if res.multi_face_landmarks:
                lms=res.multi_face_landmarks[0].landmark
                roi,box=forehead_box(fr,lms)
                if roi is not None and roi.size>0:
                    r=self.rppg.update(now,roi)
                    if r:
                        bpm_raw=r[0]; snr=r[1] if len(r)>1 else np.nan
                        bpm=float(np.clip(bpm_raw,55,120))
                        self.bpm_ma.append(bpm)
                        if len(self.bpm_ma)>10: self.bpm_ma.pop(0)
                        bpm=float(np.mean(self.bpm_ma))
                h,w,_=fr.shape
                eL=ear_from_mesh(lms,w,h,True); eR=ear_from_mesh(lms,w,h,False)
                if eL is not None and eR is not None:
                    ear=(float(eL)+float(eR))/2.0
                    self.ear_hist.append(ear)
                    mu=np.nanmedian(self.ear_hist[-60:]) if self.ear_hist else 0.28
                    sd=np.nanstd(self.ear_hist[-60:]) if self.ear_hist else 0.04
                    thr=max(0.12, mu-1.5*sd)
                    closed=ear<thr
                    if closed and not self.eye_closed:
                        self.eye_closed=True; self.eye_close_t=now
                    elif (not closed) and self.eye_closed:
                        dur=now-(self.eye_close_t or now)
                        if 0.06<=dur<=0.40 and (now-self.last_blink_t)>=0.18:
                            self.blink_count+=1; self.last_blink_t=now
                        self.eye_closed=False; self.eye_close_t=None
                    self.perclos_hist.append(1.0 if closed else 0.0)
                    blink_rate=self.blink_count*60.0/max(1.0,now-self.start_ts)
                neck=neck_proxy(fr,lms)
                self.draw.draw_landmarks(fr,res.multi_face_landmarks[0],
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.sty.get_default_face_mesh_tesselation_style())
                if box is not None:
                    x,y,w0,h0=box; cv2.rectangle(fr,(x,y),(x+w0,y+h0),(0,255,0),1)

            if not np.isnan(bpm): self.bpm_hist.append(bpm)
            if not np.isnan(snr): self.snr_hist.append(snr)
            if not np.isnan(neck): self.neck_hist.append(neck)

            btxt="--" if np.isnan(bpm) else f"{bpm:.1f}"
            stxt="--" if np.isnan(snr) else f"{snr:.2f}"
            rtxt="--" if np.isnan(blink_rate) else f"{blink_rate:.1f}"
            etxt="--" if np.isnan(ear) else f"{ear:.3f}"
            pctx="--" if not self.perclos_hist else f"{100.0*float(np.nanmean(self.perclos_hist)):.1f}%"
            ntx="--" if np.isnan(neck) else f"{neck:.2f}"
            self.tpane.setPlainText(f"BPM: {btxt}\nSNR: {stxt}\nBlink/min: {rtxt}\nEAR: {etxt}\nPERCLOS: {pctx}\nNeck(norm): {ntx}")

            draw_text(fr,f"BPM:{btxt}",12,28); draw_text(fr,f"SNR:{stxt}",12,52); draw_text(fr,f"Blink/min:{rtxt}",12,76)
            draw_text(fr,f"EAR:{etxt}",12,100); draw_text(fr,f"PERCLOS:{pctx}",12,124); draw_text(fr,f"Neck(norm):{ntx}",12,148)

            qi=to_qimage_bgr(fr)
            if qi:
                pix=QPixmap.fromImage(qi).scaled(self.vlabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.vlabel.setPixmap(pix)

            if self.frames%30==0:
                try:
                    os.makedirs("sessions",exist_ok=True)
                    live={"ts":now,"rppg_bpm":(None if np.isnan(bpm) else float(bpm)),"rppg_snr":(None if np.isnan(snr) else float(snr)),
                          "blink_per_min":(None if np.isnan(blink_rate) else float(blink_rate)),"ear":(None if np.isnan(ear) else float(ear)),
                          "perclos":(None if not self.perclos_hist else float(np.nanmean(self.perclos_hist))),
                          "neck_norm":(None if np.isnan(neck) else float(neck))}
                    with open("sessions/metrics_live.json","w") as f: json.dump(live,f)
                except: pass
        except Exception as e:
            try:
                os.makedirs("sessions",exist_ok=True)
                with open("sessions/qt_errors.log","a") as f: f.write(f"{time.time()} {repr(e)}\n")
            except: pass

    def show_results(self):
        self.res=ResultsWindow(); self.res.resize(1200,900); self.res.show(); self.res.raise_(); self.res.activateWindow()

    def closeEvent(self,e):
        try:
            if self.timer.isActive(): self.on_stop()
        except: pass
        super().closeEvent(e)

def main():
    app=QApplication(sys.argv); win=HUD(); win.resize(1320,860); win.show(); sys.exit(app.exec())

if __name__=="__main__": main()
