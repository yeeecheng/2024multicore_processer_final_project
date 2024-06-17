import cv2  

img = cv2.imread("./data/t.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 1.4)
cv2.imshow("blur_gray", blur_gray)
edges = cv2.Canny(blur_gray, 50, 64)
cv2.imshow("frame", edges)
cv2.waitKey(0)