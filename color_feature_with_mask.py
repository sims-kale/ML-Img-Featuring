import matplotlib.pyplot as plt
import numpy as np 
import cv2

sample_img = cv2.imread(r'D:\SHU\ML_lab\lab4_image_color_featuring\flower_dataset\610.jpg')
converted_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
# converted_img = cv2.cvtColor(sample_img, 4)
# cv2.imshow("sample img", sample_img)
plt.imshow(converted_img)
plt.title("Converted Image")

cv2.waitKey(0)

img_r = converted_img[:,:,0]

img_r_blur = cv2.GaussianBlur(img_r, (5,5), 0)

f, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(img_r, cmap='gray')
axes[1].imshow(img_r_blur, cmap='gray')
plt.show()

h_r_blur = cv2.calcHist([img_r_blur], [0], None, [256], [0,256])    # 0: channel, None: mask, 256: hist size, 0~256: range
plt.plot(h_r_blur)
plt.title('Histogram of the red channel of the image')
plt.show()

from cv2 import inRange
thresh_low = 100
thresh_high = 255
bw = inRange(img_r_blur, thresh_low, thresh_high)
bw = bw ==255
bw = bw.astype('uint8')
masked = bw*img_r
f,axes = plt.subplots(1,2,figsize=(10,4))
axes[0].imshow(bw, cmap='gray')
axes[1].imshow(masked, cmap ='grey')
plt.show()

h_masked = cv2.calcHist([img_r_blur], [0], bw, [256], [0,256])
plt.plot(h_masked)
plt.title("Histogram of the flower region not the background")
plt.show()

ret, thresh = cv2.threshold(img_r_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask = (thresh == 255).astype('uint8')
h_masked_r = cv2.calcHist([img_r_blur], [0], bw, [256], [0,256])
plt.plot(h_masked_r)
plt.title("The threshold value is {}".format(ret))
plt.show()
plt.imshow(mask, cmap='gray')
plt.show()

# img_r = converted_img[:,:,0]  #red channel of the image
# img_g = converted_img[:,:,1]  # Green channel of the image
# img_b = converted_img[:,:,2]  # Blue channel of the image

# h_r_object = cv2.calcHist([img_r], [0], mask, [256], [0,256])
# h_g_object = cv2.calcHist([img_g], [0],mask, [256], [0,256])
# h_b_object = cv2.calcHist([img_b], [0], mask, [256], [0,256])

hist = []
for i in range(3):
    h = cv2.calcHist([converted_img[:,:,i]], [0], mask, [256], [0,256])
    hist.append(h)
hist = np.concatenate(hist, axis=0)
plt.plot(hist)
plt.title("Concatated Histogram of the object in the image")
plt.show()

# h = [cv2.calcHist([converted_img], [i], mask, [256], [0,256]) for i in range(3)]
# f, axes = plt.subplots(1,3, figsize=(10,5))
# h - np.concatenate(h, axis=0)

