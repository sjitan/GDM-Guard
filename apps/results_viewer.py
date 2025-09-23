
import os,sys,json
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QLabel,QScrollArea,QTextEdit
from PySide6.QtGui import QPixmap,QFont
from PySide6.QtCore import Qt
class Results(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GDM-Guard — Results")
        img=QLabel(alignment=Qt.AlignCenter)
        txt=QTextEdit(); txt.setReadOnly(True); txt.setFont(QFont("Menlo",11))
        sa=QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(img)
        w=QWidget(); lay=QVBoxLayout(w); lay.addWidget(sa); lay.addWidget(txt); self.setCentralWidget(w)
        png="sessions/assessment.png"; rep="sessions/recommendation_report.txt"; js="sessions/recommendation.json"
        if os.path.exists(png): img.setPixmap(QPixmap(png).scaled(1200,800,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        else: img.setText("No plot: sessions/assessment.png")
        if os.path.exists(rep): txt.setPlainText(open(rep).read())
        elif os.path.exists(js): txt.setPlainText(open(js).read())
        else: txt.setPlainText("No recommendation")
if __name__=="__main__":
    app=QApplication(sys.argv); r=Results(); r.resize(1200,900); r.show(); sys.exit(app.exec())
