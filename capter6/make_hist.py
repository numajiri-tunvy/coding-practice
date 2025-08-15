import cv2
import numpy as np

def show_hist(hist, win_name):
    h = 255
    hist_img = np.zeros([h, hist.shape[0]], dtype=np.uint8)
    max_value = max(hist, key=lambda x: x)
    for w in range(hist.shape[0]):
        v = np.clip(np.uint8(h*hist[w]/max_value), 0, h)
        hist_img[(h - v):h,w] = 255
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, hist_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
hist: ヒストグラム(1dim)
N: ピクセル数
"""
def equalize_hist(img, hist, N):
    [height, width] = img.shape[:2]
    hist_equal_img = np.zeros([height, width], dtype=np.uint8)
    c = hist.shape[0] # ヒストグラムの高さ方向ピクセル数~=256
    cumulative_freq = np.zeros(c, dtype=np.uint32)
    # 累積分布関数を計算
    for v in range(c):
        prev = cumulative_freq[v - 1] if v > 0 else 0
        cumulative_freq[v] = (hist[v] + prev)
    # 均一化計算
    c0 = cumulative_freq[0] / N
    for h in range(height):
        for w in range(width):
            v = img[h,w]
            ev = (cumulative_freq[v]/N - c0) / (1 - c0) * 255
            hist_equal_img[h, w] = np.clip(np.uint8(ev), 0, 255)
    return hist_equal_img



def main():
    img_src = cv2.imread("src.JPG", cv2.IMREAD_GRAYSCALE)
    if(img_src is None):
        print("Failed to load image")
        return -1
    [height,width] = img_src.shape
    hist = np.zeros(256, dtype=np.int32)
    for h in range(height):
        for w in range(width):
            hist[img_src[h,w]] += 1
    
    win_src = "image"
    win_hist = "hist"
    cv2.namedWindow(win_src, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_src, img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_hist(hist, win_hist)

    hist_equal_img = equalize_hist(img_src, hist, height*width)
    print(max(hist_equal_img.flatten()))
    cv2.namedWindow("hist_equal_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("hist_equal_img", hist_equal_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()