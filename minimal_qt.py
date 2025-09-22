import sys, os, subprocess, json
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QFileDialog, QComboBox, QCheckBox
from PySide6.QtCore import Qt

HERE=Path(__file__).resolve().parent
SEED=HERE/"seeds"/"gdm_sample.mp4"
SESS=HERE/"sessions"
SESS.mkdir(exist_ok=True)
PYEXE=sys.executable

def run(cmd,wd=None):
    p=subprocess.run(cmd, cwd=wd or HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDMGuard — Minimal Demo")
        self.video=str(SEED)
        self.l_video=QLabel(self.video)
        b_pick=QPushButton("Choose Video"); b_pick.clicked.connect(self.pick_video)

        self.age=QLineEdit("30"); self.bmi=QLineEdit("26"); self.parity=QLineEdit("1")
        self.eth=QComboBox(); self.eth.addItems(["Asian","White","Black","Hispanic","Other"])
        self.prior=QCheckBox("prior_gdm"); self.prior.setChecked(True); self.family=QCheckBox("family_dm"); self.family.setChecked(True)

        b_vis=QPushButton("1) Visualize HUD"); b_vis.clicked.connect(self.do_vis)
        b_ext=QPushButton("2) Extract Features"); b_ext.clicked.connect(self.do_ext)
        b_run=QPushButton("3) Run Agent"); b_run.clicked.connect(self.do_agent)
        b_all=QPushButton("Run Demo"); b_all.clicked.connect(self.do_all)

        self.status=QLabel("Ready"); self.status.setWordWrap(True)
        self.out=QLabel(""); self.out.setAlignment(Qt.AlignLeft|Qt.AlignTop); self.out.setStyleSheet("font-family: Menlo, monospace;")

        top=QHBoxLayout()
        top.addWidget(QLabel("Video:")); top.addWidget(self.l_video,1); top.addWidget(b_pick)

        form=QHBoxLayout()
        form.addWidget(QLabel("age")); form.addWidget(self.age)
        form.addWidget(QLabel("bmi")); form.addWidget(self.bmi)
        form.addWidget(QLabel("parity")); form.addWidget(self.parity)
        form.addWidget(QLabel("eth")); form.addWidget(self.eth)
        form.addWidget(self.prior); form.addWidget(self.family)

        btns=QHBoxLayout()
        btns.addWidget(b_vis); btns.addWidget(b_ext); btns.addWidget(b_run); btns.addWidget(b_all)

        root=QVBoxLayout()
        root.addLayout(top); root.addLayout(form); root.addLayout(btns)
        root.addWidget(self.status); root.addWidget(self.out)
        self.setLayout(root)
        self.resize(900, 420)

    def pick_video(self):
        p,_=QFileDialog.getOpenFileName(self,"Pick video", str(HERE), "Video (*.mp4 *.mov)")
        if p:
            self.video=p; self.l_video.setText(self.video)

    def do_vis(self):
        self.status.setText("Visualizing…")
        rc,out=run([PYEXE,"visualize_selfie.py","--video",self.video,"--duration","10","--mirror","1","--out_json",str(SESS/"vis_metrics.json")])
        self.out.setText(out[-2000:]); self.status.setText("OK" if rc==0 else "Error")

    def do_ext(self):
        self.status.setText("Extracting…")
        rc,out=run([PYEXE,"extract_selfie_features.py","--video",self.video,"--duration","10","--out",str(SESS/"features.json")])
        self.out.setText(out[-2000:]); self.status.setText("OK" if rc==0 else "Error")

    def do_agent(self):
        self.status.setText("Scoring…")
        args=[PYEXE,"agent_cli.py","--video",self.video,
              "--age",self.age.text(),"--bmi",self.bmi.text(),"--parity",self.parity.text(),
              "--ethnicity",self.eth.currentText(),
              "--prior_gdm","1" if self.prior.isChecked() else "0",
              "--family_dm","1" if self.family.isChecked() else "0"]
        rc,out=run(args)
        self.out.setText(out[-2000:])
        self.status.setText("OK" if rc==0 else "Error")

    def do_all(self):
        self.do_vis(); self.do_ext(); self.do_agent()

def main():
    app=QApplication(sys.argv)
    w=App(); w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
