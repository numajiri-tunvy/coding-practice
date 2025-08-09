import cv2
import numpy as np

def random_dithering(img):
    [height, width] = img.shape[:2]
    dst_img = np.zeros([height, width], dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            v = img[h, w]
            r = np.clip(np.uint8(np.random.random()*255), 0, 255)
            if(v > r):
                dst_img[h, w] = 255
            else:
                dst_img[h, w] = 0
    return dst_img

def spread_error_dithering(img):
    [height, width] = img.shape[:2]
    dst_img = np.zeros([height, width], dtype=np.uint8)
    threshold = 100
    error = 0
    for h in range(height):
        for w in range(width):
            v = img[h,w] + error
            if(v > threshold):
                dst_img[h,w] = 255
                error =  v - 255
            else:
                dst_img[h,w] = 0
                error = 255 - v
    return dst_img

def ordered_dithering(img, N):
    [height, width] = img.shape[:2]
    dst_img = np.zeros([height, width], dtype=np.uint8)
    matrix = [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]
    matrix = np.array(matrix)
    matrix = matrix.reshape(N, N)
    matrix = matrix.astype(np.uint8)
    matrix *= N*N

    for h in range(height):
        for w in range(width):
            v = img[h,w]
            if(v < matrix[h%N, w%N]):
                dst_img[h,w] = 0
            else:
                dst_img[h,w] = 255
    return dst_img



def main():
    ctrl_flag = 0;
    src_img = cv2.imread("src.JPG", cv2.IMREAD_GRAYSCALE)
    if(src_img is None):
        print("Failed to load image")
        return -1;
    cv2.namedWindow("src_img", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("dst_img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("src_img", src_img)
    dst_img = ordered_dithering(src_img, 4)
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

main()