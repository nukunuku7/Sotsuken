import trimesh
import numpy as np

class ObjectLoader:
    def __init__(self):
        self.mesh = None

    def load_model(self, path):
        """
        3Dファイルを読み込んでメッシュに変換
        対応形式: STL, OBJ, PLY, GLB, GLTF, FBX
        """
        try:
            self.mesh = trimesh.load(path, force='mesh')
            if not isinstance(self.mesh, trimesh.Trimesh):
                print("警告: 複数ジオメトリ検出、最初の要素のみ使用")
                self.mesh = self.mesh.dump().geometries[0]
            return True
        except Exception as e:
            print(f"読み込みエラー: {e}")
            return False

    def get_vertices_and_faces(self):
        """
        頂点と面の配列を取得
        """
        if self.mesh:
            return self.mesh.vertices, self.mesh.faces
        return None, None

    def get_bounds(self):
        """
        モデルの外接境界ボックス(min, max)
        """
        if self.mesh:
            return self.mesh.bounds
        return None

    def get_center(self):
        """
        モデルの中心点（ジオメトリ重心）
        """
        if self.mesh:
            return self.mesh.centroid
        return None
