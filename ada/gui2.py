import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QComboBox, QAction, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint


class ImageVisualizer(QLabel):
    def __init__(self, interactive=False):
        super().__init__()
        self.interactive = interactive
        self.image = None
        self.mask = None
        self.last_strokes = []  # Stores the last strokes for undo
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = Qt.red
        self.pen_width = 5

        if interactive:
            self.setFrameStyle(QFrame.Panel | QFrame.Sunken)

    def load_image(self, image_path):
        self.image = QPixmap(image_path)
        self.mask = QPixmap(self.image.size())
        self.mask.fill(Qt.transparent)
        self.setPixmap(self.image)

    def update_display(self):
        if self.image is not None:
            painter = QPainter(self.image)
            painter.drawPixmap(0, 0, self.mask)
            painter.end()
            self.setPixmap(self.image)

    def mousePressEvent(self, event):
        if self.interactive and event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.interactive and self.drawing:
            painter = QPainter(self.mask)
            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            painter.end()
            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.interactive and event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_strokes.append(self.mask.copy())

    def undo_last_stroke(self):
        if self.last_strokes:
            self.mask = self.last_strokes.pop()
            self.update_display()


class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_paths = []
        self.current_index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Visualizer")

        # File menu
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Directory", self)
        open_action.triggered.connect(self.load_directory)
        file_menu.addAction(open_action)

        # Image Navigation Controls
        self.prev_button = QPushButton("< Previous")
        self.next_button = QPushButton("Next >")
        self.image_selector = QComboBox()
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.image_selector.currentIndexChanged.connect(self.select_image)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.image_selector)
        nav_layout.addWidget(self.next_button)

        # Image Visualizers
        self.interactive_view = ImageVisualizer(interactive=True)
        self.non_interactive_view = ImageVisualizer(interactive=False)

        # Stroke Type Buttons
        stroke_layout = QVBoxLayout()
        for i in range(1, 6):
            btn = QPushButton(f"B{i}")
            stroke_layout.addWidget(btn)

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_layout)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.interactive_view)
        image_layout.addLayout(stroke_layout)
        image_layout.addWidget(self.non_interactive_view)
        main_layout.addLayout(image_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                                f.endswith(('.png', '.jpg', '.jpeg'))]
            if self.image_paths:
                self.image_selector.clear()
                self.image_selector.addItems([os.path.basename(p) for p in self.image_paths])
                self.load_image(0)

    def load_image(self, index):
        if self.image_paths:
            self.current_index = index
            img_path = self.image_paths[index]
            self.interactive_view.load_image(img_path)
            self.non_interactive_view.load_image(img_path)

    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.load_image(self.current_index - 1)
            self.image_selector.setCurrentIndex(self.current_index)

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.load_image(self.current_index + 1)
            self.image_selector.setCurrentIndex(self.current_index)

    def select_image(self, index):
        self.load_image(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())
