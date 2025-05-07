import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QPoint
from PIL import Image
import numpy as np

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.start_point = None
        self.end_point = None
        self.rubber_band_rect = None
        self.image = None  # PIL Image
        self.display_image = None  # Resized PIL Image
        self.scale_factor = 1.0

    def load_image(self, path):
        self.image = Image.open(path)
        # Resize for visualization
        max_dim = 800  # max display dimension
        width, height = self.image.size
        if max(width, height) > max_dim:
            self.scale_factor = max_dim / max(width, height)
            new_size = (int(width * self.scale_factor), int(height * self.scale_factor))
            self.display_image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            self.scale_factor = 1.0
            self.display_image = self.image.copy()
        # Convert to QPixmap
        data = self.display_image.convert("RGBA").tobytes("raw", "RGBA")
        qimg = QImage(data, self.display_image.width, self.display_image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)
        self.adjustSize()
        self.start_point = None
        self.end_point = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            self.end_point = event.pos()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect.normalized())

    def get_crop_rect(self):
        if self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            x1 = int(rect.x() / self.scale_factor)
            y1 = int(rect.y() / self.scale_factor)
            x2 = int((rect.x() + rect.width()) / self.scale_factor)
            y2 = int((rect.y() + rect.height()) / self.scale_factor)
            return (x1, y1, x2, y2)
        return None

    def save_cropped_image(self, save_path):
        rect = self.get_crop_rect()
        if rect and self.image:
            x1, y1, x2, y2 = rect
            cropped = self.image.crop((x1, y1, x2, y2))
            cropped.save(save_path, format="PNG")
            print(f"Cropped image saved to {save_path}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Map Cropping Tool")
        self.image_label = ImageLabel()

        load_button = QPushButton("Load TIF")
        load_button.clicked.connect(self.load_image)

        save_button = QPushButton("Save Crop")
        save_button.clicked.connect(self.save_crop)

        layout = QVBoxLayout()
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TIF", "", "TIFF Files (*.tif *.tiff)")
        if path:
            self.image_label.load_image(path)

    def save_crop(self):
        if not self.image_label.start_point or not self.image_label.end_point:
            print("No crop selected!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Cropped Image", "", "PNG Files (*.png)")
        if path:
            self.image_label.save_cropped_image(path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
