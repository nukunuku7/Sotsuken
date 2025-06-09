import cv2
import numpy as np

class MediaPlayer:
    def __init__(self, warp_func=None):
        """
        Parameters:
        - warp_func: 画像補正関数（画像を受け取り、補正済み画像を返す）
        """
        self.warp_func = warp_func if warp_func else (lambda img: img)

    def play_image(self, path):
        image = cv2.imread(path)
        if image is None:
            print(f"画像読み込み失敗: {path}")
            return
        corrected = self.warp_func(image)
        cv2.imshow("補正画像", corrected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def play_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"動画読み込み失敗: {path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            corrected = self.warp_func(frame)
            cv2.imshow("補正動画", corrected)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def capture_window(self, bbox):
        """
        bbox: (x, y, width, height) で指定された画面領域をキャプチャ
        """
        import mss
        with mss.mss() as sct:
            monitor = {"top": bbox[1], "left": bbox[0], "width": bbox[2], "height": bbox[3]}
            img = np.array(sct.grab(monitor))[:, :, :3]  # BGRに変換
            corrected = self.warp_func(img)
            cv2.imshow("ウィンドウキャプチャ補正", corrected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
