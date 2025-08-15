"""
opencvの動作確認用スクリプト
黒い画像がsrcとして表示される

opencvのインストール
pip install opencv-python

opencvのバージョン確認
python -c "import cv2; print(cv2.__version__)"


"""

import cv2
import numpy as np

img = np.zeros((480, 640), np.uint8);
cv2.imshow('src', img);
cv2.waitKey(0);
cv2.destroyAllWindows();

print(f"opencv version: {cv2.__version__}")