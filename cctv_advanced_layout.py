import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, 
                             QSplitter, QLabel, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from datetime import datetime

class VideoThread(QThread):
    """웹캠 영상을 백그라운드에서 읽어오는 스레드"""
    change_pixmap_signal = pyqtSignal(QImage)

    def run(self):
        # 0번은 기본 웹캠입니다.
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # BGR을 RGB로 변환
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(1280, 720, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            else:
                break
        cap.release()

class CameraWidget(QLabel):
    def __init__(self, name, color="#222"):
        super().__init__()
        self.name = name
        self.setText(name)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background-color: {color}; color: white; border: 2px solid #444; border-radius: 5px;")
        self.setMinimumSize(100, 80)

class AdvancedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # 시계 타이머
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        # 웹캠 스레드 시작
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def setup_ui(self):
        self.setWindowTitle("Nova Professional CCTV Monitor - Webcam Test")
        self.resize(1280, 800)
        self.setStyleSheet("background-color: #111; color: #eee;")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 좌측 패널
        self.left_panel = QFrame()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.cam_list = QListWidget()
        for i in range(1, 17): self.cam_list.addItem(f"Camera {i:02d} (Live)")
        self.left_layout.addWidget(QLabel("CAMERA LIST"))
        self.left_layout.addWidget(self.cam_list)
        self.splitter.addWidget(self.left_panel)
        
        # 우측 L-Layout
        self.video_area = QWidget()
        self.video_layout = QGridLayout(self.video_area)
        self.setup_l_layout()
        self.splitter.addWidget(self.video_area)
        
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 8)
        self.main_layout.addWidget(self.splitter)

    def setup_l_layout(self):
        # 메인 집중 뷰 (웹캠이 나올 자리)
        self.main_cam = CameraWidget("WEBCAM FEED", "#000")
        self.main_cam.setStyleSheet(self.main_cam.styleSheet() + "border-color: #0078d7;")
        self.video_layout.addWidget(self.main_cam, 0, 0, 3, 3)
        
        # 서브 뷰들 (정적 위젯)
        for i in range(3): self.video_layout.addWidget(CameraWidget(f"Sub {i+1}"), i, 3, 1, 1)
        for i in range(4): self.video_layout.addWidget(CameraWidget(f"Sub {i+4}"), 3, i, 1, 1)

    def update_image(self, cv_img):
        """웹캠 프레임을 메인 위젯에 업데이트"""
        self.main_cam.setPixmap(QPixmap.fromImage(cv_img))

    def update_clock(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.setWindowTitle(f"Nova CCTV Monitor - {current_time}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedViewer()
    window.show()
    sys.exit(app.exec())
