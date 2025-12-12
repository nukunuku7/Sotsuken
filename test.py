import sys
sys.path.pop(0)  # カレントディレクトリを import path から除去
import cv2

print(cv2.getBuildInformation())
