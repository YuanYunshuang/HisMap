import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QRect


class BoundingBoxVisualizer(QLabel):
    def __init__(self, image_path, bounding_boxes, parent=None):
        super().__init__(parent)
        self.image = QPixmap(image_path)
        self.bounding_boxes = bounding_boxes  # List of bounding boxes [(x, y, w, h, color)]
        self.selected_color = None
        self.selected_bbox = None
        self.setPixmap(self.image)
        self.setFixedSize(self.image.size())

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

        for bbox in self.bounding_boxes:
            x, y, w, h, color = bbox
            pen = QPen(QColor(color))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(QRect(x, y, w, h))

    def set_selected_color(self, color):
        self.selected_color = color
        cursor_pixmap = QPixmap(16, 16)
        cursor_pixmap.fill(QColor(color))
        cursor = QCursor(cursor_pixmap)
        self.setCursor(cursor)

    def mousePressEvent(self, event):
        if not self.selected_color:
            return

        x, y = event.pos().x(), event.pos().y()
        for i, (bx, by, bw, bh, _) in enumerate(self.bounding_boxes):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.bounding_boxes[i] = (bx, by, bw, bh, self.selected_color)
                self.update()
                break


class ColorPicker(QWidget):
    def __init__(self, visualizer, parent=None):
        super().__init__(parent)
        self.visualizer = visualizer
        self.layout = QHBoxLayout()
        self.colors = ["red", "green", "blue", "yellow"]

        for color in self.colors:
            button = QPushButton()
            button.setStyleSheet(f"background-color: {color}")
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _, c=color: self.visualizer.set_selected_color(c))
            self.layout.addWidget(button)

        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bounding Box Visualizer")

        image_path = "sample_image.jpg"  # Change to your image path
        bounding_boxes = [(50, 50, 100, 100, "black"), (200, 150, 80, 120, "black")]

        self.visualizer = BoundingBoxVisualizer(image_path, bounding_boxes)
        self.color_picker = ColorPicker(self.visualizer)

        layout = QVBoxLayout()
        layout.addWidget(self.visualizer)
        layout.addWidget(self.color_picker)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
