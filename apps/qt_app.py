import sys, time, cv2, numpy as np, mediapipe as mp
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from lib.rppg_utils import RPPG, forehead_roi, ear_from_mesh, neck_proxy
import os,sys; sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

DU = mp.solutions.drawing_utils
DS = mp.solutions.drawing_styles
FM_CONTOURS = mp.solutions.face_mesh.FACEMESH_TESSELATION

class App(QMainWindow):
    def __init__(self, video, mirror=False, hud_scale=1.2):
        super().__init__()
        self.setWindowTitle("GDMGuard — Selfie Telemetry HUD")
        self.lab = QLabel(); self.lab.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(); lay.addWidget(self.lab)
        w = QWidget(); w.setLayout(lay); self.setCentralWidget(w)

        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened(): raise SystemExit("video_open_failed")

        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, refine_landmarks=True,
            max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.rppg = RPPG(min_sec=8.0, max_sec=20.0)
        self.mirror = mirror
        self.hud = float(hud_scale)

        self.t0 = time.time()
        self.blinks = 0
        self.prev_open = True
        self.perclos_vals, self.necks, self.hrs, self.snrs = [], [], [], []

        self.timer = QTimer(); self.timer.timeout.connect(self.step); self.timer.start(30)

    def step(self):
        ok, fr = self.cap.read()
        if not ok:
            self.timer.stop(); self.cap.release(); return

        if self.mirror: fr = cv2.flip(fr, 1)
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        h, w = fr.shape[:2]

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0]
            DU.draw_landmarks(
                image=fr,
                landmark_list=lms,
                connections=FM_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DS.get_default_face_mesh_tesselation_style()
            )
            eL = ear_from_mesh(lms.landmark, w, h, True)
            eR = ear_from_mesh(lms.landmark, w, h, False)
            openv = 1.0 if 0.5 * (eL + eR) > 0.25 else 0.0
            self.perclos_vals.append(1.0 - openv)
            if self.prev_open and openv < 0.5: self.blinks += 1
            self.prev_open = openv >= 0.5

            npx = neck_proxy(fr, lms.landmark); self.necks.append(npx)

            roi = forehead_roi(fr, lms.landmark)
            if roi is not None:
                roi_img, (x1, y1, x2, y2) = roi
                est = self.rppg.update(time.time(), roi_img)
                if est is not None:
                    hr, snr = est; self.hrs.append(hr); self.snrs.append(snr)
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        dur = max(1e-6, time.time() - self.t0)
        perclos = float(np.clip(np.mean(self.perclos_vals) if self.perclos_vals else 0.2, 0, 1))
        br = float(self.blinks / dur * 60.0)
        neck = float(np.median(self.necks) if self.necks else 1.0)
        hr = float(np.median(self.hrs) if self.hrs else 0.0)
        snr = float(np.median(self.snrs) if self.snrs else 0.0)

        s = self.hud
        cv2.putText(fr, f"HR {hr:.0f} bpm   SNR {snr:.1f}", (20, int(40*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, (255,255,255), 2)
        cv2.putText(fr, f"PERCLOS {perclos:.2f}   Blinks/min {br:.0f}", (20, int(80*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, (255,255,255), 2)
        cv2.putText(fr, f"Neck_norm {neck:.2f}", (20, int(120*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9*s, (255,255,255), 2)

        disp = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], 3*disp.shape[1], QImage.Format_RGB888)
        self.lab.setPixmap(QPixmap.fromImage(qimg))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--hud_scale", type=float, default=1.2)
    a = ap.parse_args()
    app = QApplication(sys.argv)
    w = App(a.video, mirror=bool(a.mirror), hud_scale=a.hud_scale); w.resize(1100, 720); w.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()