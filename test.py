import os
import ctypes

opencv_bin = r"C:\Users\vrlab\.vscode\nukunuku\OpenCV_CUDAbuild\opencv\x64\vc17\bin"
os.environ['PATH'] = opencv_bin + ";" + os.environ['PATH']

# OpenCV DLL のロード確認
opencv_dll = os.path.join(opencv_bin, "opencv_world4130.dll")
ctypes.WinDLL(opencv_dll)

# cv2 のインポート
import cv2
print(cv2.getBuildInformation())
