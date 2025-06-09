from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QBrush, QPen, QPainter
import json
import os

class GridEditorDialog(QDialog):
    def __init__(self, display_name, config=None, screen_geometry=None):
        super().__init__()
        self.display_name = display_name
        self.setWindowTitle(f"{display_name} の端点編集")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: black;")
        self.resize(screen_geometry.width(), screen_geometry.height())

        self.scene = QGraphicsScene(0, 0, screen_geometry.width(), screen_geometry.height())
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

        self.rows, self.cols = 10, 10
        self.grid_points = self.load_or_create_grid(config, screen_geometry)
        self.draw_grid()

        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("background-color: #444444; color: white;")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def load_or_create_grid(self, config, screen_geometry):
        if config:
            return [[tuple(pt) for pt in row] for row in config]
        return self.create_default_grid(screen_geometry)

    def create_default_grid(self, screen_geometry):
        width = screen_geometry.width()
        height = screen_geometry.height()
        grid = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                x = width * j / (self.cols - 1)
                y = height * i / (self.rows - 1)
                row.append((x, y))
            grid.append(row)
        return grid

    def draw_grid(self):
        self.points = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        grid_pen = QPen(Qt.green)
        grid_pen.setWidth(2)

        for i in range(self.rows):
            for j in range(self.cols):
                is_edge = i in [0, self.rows - 1] or j in [0, self.cols - 1]
                if is_edge:
                    x, y = self.grid_points[i][j]
                    point = EditablePoint(x, y, 5)
                    self.scene.addItem(point)
                    self.points[i][j] = point

        for i in range(self.cols - 1):
            self.add_line(0, i, 0, i + 1, grid_pen)
            self.add_line(self.rows - 1, i, self.rows - 1, i + 1, grid_pen)
        for i in range(self.rows - 1):
            self.add_line(i, 0, i + 1, 0, grid_pen)
            self.add_line(i, self.cols - 1, i + 1, self.cols - 1, grid_pen)

    def add_line(self, i1, j1, i2, j2, pen):
        x1, y1 = self.grid_points[i1][j1]
        x2, y2 = self.grid_points[i2][j2]
        self.scene.addLine(x1, y1, x2, y2, pen)

    def get_current_config(self):
        config = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                point = self.points[i][j]
                if point:
                    pos = point.scenePos()
                    config[i][j] = (pos.x(), pos.y())
                else:
                    config[i][j] = self.grid_points[i][j]
        return config

class EditablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setBrush(QBrush(Qt.red))
        self.setPen(QPen(Qt.red))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setPos(x, y)

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)