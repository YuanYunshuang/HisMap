import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QHBoxLayout, QFileDialog, QComboBox, QAction, QToolBar
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QRect


class BoundingBoxVisualizer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QPixmap()
        self.bounding_boxes = []  # List of bounding boxes [(x, y, w, h, color)]
        self.selected_color = None
        self.setFixedSize(600, 400)  # Default size

    def load_image(self, image_path):
        self.image = QPixmap(image_path)
        self.setPixmap(self.image)
        self.setFixedSize(self.image.size())
        self.bounding_boxes = []  # Reset bounding boxes when loading a new image
        self.update()

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
        self.layout = QVBoxLayout()

        # Label for the color picker
        self.label = QLabel("Select Bounding Box Color:")
        self.layout.addWidget(self.label)

        # Buttons for color selection
        button_layout = QHBoxLayout()
        self.colors = ["red", "green", "blue", "yellow"]

        for color in self.colors:
            button = QPushButton()
            button.setStyleSheet(f"background-color: {color}")
            button.setText(color)
            button.setFixedSize(80, 40)
            button.clicked.connect(lambda _, c=color: self.visualizer.set_selected_color(c))
            button_layout.addWidget(button)

        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bounding Box Visualizer")

        # Initialize components
        self.visualizer = BoundingBoxVisualizer()
        self.color_picker = ColorPicker(self.visualizer)

        # Image directory management
        self.image_directory = None
        self.image_list = []
        self.current_index = -1

        # Setup layout
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.visualizer)
        layout.addWidget(self.color_picker)

        # Navigation toolbar
        self.toolbar = QToolBar("Navigation")
        self.addToolBar(self.toolbar)

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.image_selector = QComboBox()

        self.prev_button.clicked.connect(self.load_previous_image)
        self.next_button.clicked.connect(self.load_next_image)
        self.image_selector.currentIndexChanged.connect(self.change_image)

        self.toolbar.addWidget(self.prev_button)
        self.toolbar.addWidget(self.next_button)
        self.toolbar.addWidget(self.image_selector)

        # Main container
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Directory", self)
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)

    def open_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.image_directory = directory
            self.image_list = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_list.sort()

            if self.image_list:
                self.current_index = 0
                self.image_selector.clear()
                self.image_selector.addItems(self.image_list)
                self.load_image_at_index(self.current_index)

    def load_image_at_index(self, index):
        if 0 <= index < len(self.image_list):
            image_path = os.path.join(self.image_directory, self.image_list[index])
            self.visualizer.load_image(image_path)
            self.current_index = index
            self.image_selector.setCurrentIndex(index)

    def load_previous_image(self):
        if self.current_index > 0:
            self.load_image_at_index(self.current_index - 1)

    def load_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.load_image_at_index(self.current_index + 1)

    def change_image(self, index):
        if index != self.current_index:
            self.load_image_at_index(index)


# Run Application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
