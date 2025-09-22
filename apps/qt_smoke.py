
import sys, os, cv2, numpy as np
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog
from PySide6.QtCore import QTimer,Qt
from PySide6.QtGui import QImage,QPixmap
import argparse

def to_qimage(frame):
    if frame is None: return None
    if frame.ndim==2:
        h,w=frame.shape
        return QImage(frame.data,w,h,w,QImage.Format_Grayscale8).copy()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    return QImage(rgb.data,w,h,3*w,QImage.Format_RGB888).copy()

class Viewer(QMainWindow):
    def __init__(self,video=None,mirror=1):
        super().__init__()
        self.setWindowTitle("GDMGuard Qt Smoke Test")
        self.label=QLabel("Ready"); self.label.setAlignment(Qt.AlignCenter)
        self.status=QLabel("Idle")
        self.btnStart=QPushButton("Start"); self.btnStop=QPushButton("Stop")
        self.btnOpen=QPushButton("Open File"); self.btnQuit=QPushButton("Quit")
        row=QHBoxLayout()
        for b in (self.btnOpen,self.btnStart,self.btnStop,self.btnQuit,self.status): row.addWidget(b)
        lay=QVBoxLayout(); lay.addLayout(row); lay.addWidget(self.label)
        w=QWidget(); w.setLayout(lay); self.setCentralWidget(w)
        self.timer=QTimer(); self.timer.timeout.connect(self.on_frame)
        self.cap=None; self.path=video; self.mirror=bool(mirror)
        self.btnOpen.clicked.connect(self.on_open)
        self.btnStart.clicked.connect(self.on_start)
        self.btnStop.clicked.connect(self.on_stop)
        self.btnQuit.clicked.connect(self.close)
    def on_open(self):
        p,_=QFileDialog.getOpenFileName(self,"Choose video","","Video Files (*.mp4 *.mov *.m4v);;All Files (*)")
        if p: self.path=p; self.status.setText(os.path.basename(p))
    def open_cap(self):
        if self.cap and self.cap.isOpened(): return True
        if self.path and os.path.exists(self.path): self.cap=cv2.VideoCapture(self.path)
        else: self.cap=cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)
        ok=bool(self.cap and self.cap.isOpened())
        self.status.setText("Video" if (self.path and os.path.exists(self.path)) else ("Camera OK" if ok else "Open failed"))
        return ok
    def on_start(self):
        if not self.open_cap(): return
        self.timer.start(30); self.status.setText("Playing")
    def on_stop(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened(): self.cap.release()
        self.status.setText("Stopped")
    def on_frame(self):
        if not (self.cap and self.cap.isOpened()): return
        ok,fr=self.cap.read()
        if not ok: self.on_stop(); return
        if self.mirror: fr=cv2.flip(fr,1)
        qi=to_qimage(fr)
        if qi: self.label.setPixmap(QPixmap.fromImage(qi))
    def closeEvent(self,e):
        self.on_stop(); return super().closeEvent(e)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",default="seeds/gdm_sample.mp4")
    ap.add_argument("--mirror",type=int,default=1)
    args=ap.parse_args()
    app=QApplication(sys.argv)
    win=Viewer(video=args.video,mirror=args.mirror)
    win.resize(1060,720); win.show()
    sys.exit(app.exec())

if __name__=="__main__": main()
