#Task 1: RGB Color Space

import cv2
import matplotlib.pyplot as plt
sample_img = cv2.imread(r'D:\SHU\ML_lab\lab4_image_color_featuring\sample_flower.jpg')
# converted_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
converted_img = cv2.cvtColor(sample_img, 4)
cv2.imshow("sample", sample_img)
cv2.imshow("converted img", converted_img)
cv2.waitKey(0)
exit()



import numpy as np
img_shape= converted_img.shape
print("Image Shape ", img_shape)

red_channel = sample_img[:, :, 0]
green_channel = sample_img[:, :, 1]
blue_channel = sample_img[:, :, 2]
cv2.imshow("red channel", red_channel)
cv2.imshow("green channel", green_channel)
cv2.imshow("blue channel", blue_channel)
# plt.imshow(red_channel)
# plt.show()
cv2.waitKey(0)


#Task 2: Display RGB Channels  

# plt.figure
fig,axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(red_channel, cmap='Reds')
axes[0].set_title('Red Channel')
axes[1].imshow(green_channel, cmap='Greens')
axes[1].set_title('Green Channel')
axes[2].imshow(blue_channel, cmap='Blues')
axes[2].set_title('Blue Channel')
plt.show()
cv2.waitKey(0)


