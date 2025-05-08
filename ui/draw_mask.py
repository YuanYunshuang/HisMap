import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction, QPushButton, QVBoxLayout, QWidget
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QColor, QImage, QPolygonF
)
from PyQt5.QtCore import Qt, QPointF, QPoint
import numpy as np
from PIL import Image

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.scale = 1.0
        self.polygons = []
        self.current_polygon = []
        self.setMouseTracking(True)

        self.offset = QPointF(0, 0)
        self.last_mouse_pos = None
        self.panning = False

    def load_image(self, filename):
        self.image = QImage(filename)
        self.pixmap = QPixmap.fromImage(self.image)
        self.scale = 1.0
        self.polygons.clear()
        self.current_polygon.clear()
        self.update()

    def paintEvent_(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(0, 0, self.pixmap)

            # Draw finished polygons
            for poly in self.polygons:
                painter.setPen(QPen(Qt.red, 2))
                painter.setBrush(QColor(255, 0, 0, 127))
                painter.drawPolygon(QPolygonF(poly))

            # Draw current polygon with transparency
            if len(self.current_polygon) > 1:
                painter.setPen(QPen(Qt.green, 2, Qt.DashLine))
                painter.setBrush(QColor(0, 255, 0, 127))
                painter.drawPolygon(QPolygonF(self.current_polygon))
            elif len(self.current_polygon) == 1:
                # Draw single point
                painter.setPen(QPen(Qt.green, 5))
                painter.drawPoint(self.current_polygon[0])

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.translate(self.offset)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(0, 0, self.pixmap)

            # Draw polygons relative to offset + scale
            painter.translate(0, 0)  # no extra translation needed

            for poly in self.polygons:
                painter.setPen(QPen(Qt.red, 2))
                painter.setBrush(QColor(255, 0, 0, 127))
                painter.drawPolygon(QPolygonF(poly))

            if len(self.current_polygon) > 1:
                painter.setPen(QPen(Qt.green, 2, Qt.DashLine))
                painter.setBrush(QColor(0, 255, 0, 127))
                painter.drawPolygon(QPolygonF(self.current_polygon))
            elif len(self.current_polygon) == 1:
                painter.setPen(QPen(Qt.green, 5))
                painter.drawPoint(self.current_polygon[0])

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        factor = 1.1 if angle > 0 else 0.9
        self.scale *= factor
        self.update()

    def mousePressEvent_(self, event):
        pos = event.pos() / self.scale
        if event.button() == Qt.LeftButton:
            self.current_polygon.append(QPointF(pos))
            self.update()
        elif event.button() == Qt.RightButton and len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon.clear()
            self.update()

    def mousePressEvent(self, event):
        pos = (event.pos() - self.offset) / self.scale
        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.NoModifier:  # no modifier: add point
                self.current_polygon.append(QPointF(pos))
                self.update()
            else:
                # left-click + any modifier → fallback to point adding
                pass
        elif event.button() == Qt.RightButton and len(self.current_polygon) == 0:
            self.last_mouse_pos = event.pos()
            self.panning = True
        elif event.button() == Qt.RightButton and len(self.current_polygon) >= 3:
            self.polygons.append(list(self.current_polygon))
            self.current_polygon.clear()
            self.update()

    def mouseMoveEvent(self, event):
        if self.panning and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.last_mouse_pos = None

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier) or event.key() == Qt.Key_Backspace:
            if self.current_polygon:
                self.current_polygon.pop()
            elif self.polygons:
                self.polygons.pop()
            self.update()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_mask()

    def save_mask(self):
        if not self.image:
            return
        width = self.image.width()
        height = self.image.height()
        mask = np.zeros((height, width), dtype=np.uint8)

        from PyQt5.QtGui import QPainterPath
        from PyQt5.QtGui import QRegion

        for poly in self.polygons:
            path = QPainterPath()
            path.moveTo(poly[0])
            for point in poly[1:]:
                path.lineTo(point)
            path.closeSubpath()

            region = QRegion(path.toFillPolygon().toPolygon())
            for y in range(height):
                for x in range(width):
                    if region.contains(QPoint(x, y)):  # ✅ FIXED HERE
                        mask[y, x] = 255

        filename, _ = QFileDialog.getSaveFileName(self, "Save Mask", "", "PNG Files (*.png)")
        if filename:
            im = Image.fromarray(mask)
            im.save(filename)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.viewer = ImageViewer()

        open_action = QAction("&Open Image", self)
        open_action.triggered.connect(self.open_image)

        save_action = QAction("&Save Mask (Ctrl+S)", self)
        save_action.triggered.connect(self.viewer.save_mask)

        undo_action = QAction("&Undo (Ctrl+Z)", self)
        undo_action.triggered.connect(self.undo_point)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(undo_action)

        btn_undo = QPushButton("Undo Last Point")
        btn_undo.clicked.connect(self.undo_point)
        btn_save = QPushButton("Save Mask")
        btn_save.clicked.connect(self.viewer.save_mask)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(btn_undo)
        layout.addWidget(btn_save)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Polygon Mask Drawer")
        self.resize(800, 800)

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if filename:
            self.viewer.load_image(filename)

    def undo_point(self):
        if self.viewer.current_polygon:
            self.viewer.current_polygon.pop()
        elif self.viewer.polygons:
            self.viewer.polygons.pop()
        self.viewer.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
