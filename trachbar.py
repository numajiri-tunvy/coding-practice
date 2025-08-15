import cv2
import numpy as np

def noting(x):
    pass

def main():
    src_img = cv2.imread("src.JPG", cv2.IMREAD_COLOR)
    if(src_img is None):
        print("Failed to load image")
        return -1;
    cv2.namedWindow("src_img", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("dst_img", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("gamma", "dst_img", 1, 10, noting)

    cv2.imshow("src_img", src_img)

    while(True):
        gamma = cv2.getTrackbarPos("gamma", "dst_img") + 1.0

        lut = np.array(range(256), dtype=np.uint8)
        lut = np.clip(np.uint8(255*((lut/255)**(1/gamma))), 0, 255)
        dst_img = cv2.LUT(src_img, lut)
        cv2.imshow("dst_img", dst_img)
        
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break
    cv2.destroyAllWindows()
    return 0

main()