from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QBrush, QPen, QPainter

class GridEditorDialog(QDialog):
    def __init__(self, display_name, config=None, screen_geometry=None):
        super().__init__()
        self.setWindowTitle(f"{display_name} の端点編集")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: black;") # 黒に近いグレー
        self.resize(screen_geometry.width(), screen_geometry.height())
        
        self.scene = QGraphicsScene(0, 0, screen_geometry.width(), screen_geometry.height())
        self.view = ZoomableGraphicsView()
        self.view.setScene(self.scene)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

        self.grid_points = self.create_default_grid(screen_geometry)
        self.draw_grid()

        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("background-color: #444444; color: white;")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def create_default_grid(self, screen_geometry, rows=10, cols=10):
        width = screen_geometry.width()
        height = screen_geometry.height()
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                x = width * j / (cols - 1)
                y = height * i / (rows - 1)
                row.append((x, y))
            grid.append(row)
        return grid

    def draw_grid(self):
        rows, cols = 10, 10
        self.points = [[None for _ in range(cols)] for _ in range(rows)]

        # 緑色のペン（太さ2）
        grid_pen = QPen(Qt.green)
        grid_pen.setWidth(2)

        for i in range(rows):
            for j in range(cols):
                is_edge = i in [0, rows - 1] or j in [0, cols - 1]
                if is_edge:
                    x, y = self.grid_points[i][j]
                    point = EditablePoint(x, y, 5)
                    self.scene.addItem(point)
                    self.points[i][j] = point

        # 横方向のグリッド線（上下端）
        for i in range(cols - 1):
            x1, y1 = self.grid_points[0][i]
            x2, y2 = self.grid_points[0][i + 1]
            self.scene.addLine(x1, y1, x2, y2, grid_pen)

            x1, y1 = self.grid_points[-1][i]
            x2, y2 = self.grid_points[-1][i + 1]
            self.scene.addLine(x1, y1, x2, y2, grid_pen)

        # 縦方向のグリッド線（左右端）
        for i in range(rows - 1):
            x1, y1 = self.grid_points[i][0]
            x2, y2 = self.grid_points[i + 1][0]
            self.scene.addLine(x1, y1, x2, y2, grid_pen)

            x1, y1 = self.grid_points[i][cols - 1]
            x2, y2 = self.grid_points[i + 1][cols - 1]
            self.scene.addLine(x1, y1, x2, y2, grid_pen)

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
