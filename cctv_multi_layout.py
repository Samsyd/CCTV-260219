import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class CameraWidget(QLabel):
    """개별 카메라 영상을 표시하는 위젯"""
    def __init__(self, name):
        super().__init__()
        self.setText(name)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")
        self.setMinimumSize(160, 120)

class MultiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nova CCTV Multi-Layout Viewer")
        self.resize(1024, 768)

        # 메인 레이아웃 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 상단 제어 버튼부
        self.button_layout = QHBoxLayout()
        btn2x2 = QPushButton("2 x 2 View")
        btn3x3 = QPushButton("3 x 3 View")
        btn4x4 = QPushButton("4 x 4 View")
        
        btn2x2.clicked.connect(lambda: self.change_layout(2))
        btn3x3.clicked.connect(lambda: self.change_layout(3))
        btn4x4.clicked.connect(lambda: self.change_layout(4))

        self.button_layout.addWidget(btn2x2)
        self.button_layout.addWidget(btn3x3)
        self.button_layout.addWidget(btn4x4)
        self.main_layout.addLayout(self.button_layout)

        # 카메라 그리드 레이아웃
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.main_layout.addWidget(self.grid_container)

        self.change_layout(2) # 기본 2x2 시작

    def clear_grid(self):
        """기존 그리드 위젯들 제거"""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def change_layout(self, size):
        """그리드 크기 변경 (2x2, 3x3, 4x4)"""
        self.clear_grid()
        for row in range(size):
            for col in range(size):
                cam_idx = row * size + col + 1
                cam_widget = CameraWidget(f"Camera {cam_idx}\n(Connecting...)")
                self.grid_layout.addWidget(cam_widget, row, col)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiViewer()
    window.show()
    sys.exit(app.exec())
