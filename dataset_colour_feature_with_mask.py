import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
from cv2 import inRange
import numpy as np
from sklearn.preprocessing import MinMaxScaler




def display_grid(images, cols=7, figsize=(13, 7), cmap=None):
    """Displays a list of images in a grid."""
    num_images = len(images)
    rows = (num_images // cols) + (num_images % cols > 0)  # Auto-calculate rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i], cmap=cmap)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def display_histogram_grid(histograms, cols=7, title="Histogram"):
    """ Display histograms in a grid format """
    num_hist = len(histograms)
    rows = (num_hist // cols) + (num_hist % cols > 0)
    # plt.title("Histogram of the flower region not the background")
    
    fig, axes = plt.subplots(rows, cols, figsize=(13, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes.flatten()):
        if i < len(histograms):
            hist = histograms[i].flatten()
            hist_norm = hist / np.max(hist) if np.max(hist) != 0 else hist
            ax.plot(hist_norm, color='blue')
            ax.set_xlim([0, 256])
            ax.set_ylim([0, 1])  # Normalized range
            axes[i].set_yticks([0.0, 0.5,1.0])  # ✅ Set y-ticks
            axes[i].set_xticks([])  # Hide x-ticks for clarity
        else:
            ax.axis("off")  # Hide empty subplots
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def display_OTSU_histogram(histograms, cols=7, title=" OTSU Histogram"):
    """ Display histograms in a grid format """
    num_hist = len(histograms)
    rows = (num_hist // cols) + (num_hist % cols > 0)
    # plt.title("Histogram of the flower region not the background")
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes.flatten()):
        if i < len(histograms):
            hist = histograms[i].flatten()
            hist_norm = hist / np.max(hist) if np.max(hist) != 0 else hist
            ax.plot(hist_norm, color='blue')
            ax.set_xlim([0, 256])
            ax.set_ylim([0, 1])
            axes[i].set_yticks([0.0, 0.5,1.0])  
            axes[i].set_xticks([])  
        else:
            ax.axis("off")  # Hide empty subplots
    plt.suptitle(title, fontsize=14)
    # plt.tight_layout()
    plt.show()

def apply_threshold(images, thresh_low=100, thresh_high=255):
    """Applies binary thresholding to a list of images."""
    thresholded_images = []
    masked_images = []

    for img in images:
        bw = cv2.inRange(img, thresh_low, thresh_high)  # Thresholding
        bw = bw == 255  # Convert to boolean mask
        bw = bw.astype('uint8')  # Convert back to uint8
        masked = bw * img  # Apply mask to original image
        thresholded_images.append(bw)
        masked_images.append(masked)

    return thresholded_images, masked_images

def display_threshold_grid(images, cols=7, cmap="gray", title="Images Grid"):
    """Displays a list of images in a grid format."""
    num_images = len(images)
    rows = (num_images // cols) + (num_images % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i], cmap=cmap)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide extra subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def get_hist(sample_img, mask):
    img_r_blur = cv2.GaussianBlur(sample_img,(5,5),0)
    hist = []
    for i in range(3):
        h = cv2.calcHist([img_r_blur],[i],mask,[256],[0,256])
        hist.append(h)
    hist = np.concatenate(hist, axis=0) # use all the R,G,B histogram and connect them into one bigger histogram
    scaler = MinMaxScaler()
    hist_scale = scaler.fit_transform(hist.reshape(-1,1))
    return hist_scale







# sample_img = cv2.imread(r'D:\SHU\ML_lab\Lab5_img_color_featuring_with_mask\flower_dataset\641.jpg')
# img_r = converted_img[:,:,0]
# img_r_blur = cv2.GaussianBlur(img_r, (5,5), 0)

# f, axes = plt.subplots(1,2, figsize=(10,5))
# axes[0].imshow(img_r, cmap='gray')
# axes[1].imshow(img_r_blur, cmap='gray')
# plt.show()

# h_r_blur = cv2.calcHist([img_r_blur], [0], None, [256], [0,256])    # 0: channel, None: mask, 256: hist size, 0~256: range
# plt.plot(h_r_blur)
# plt.title('Histogram of the red channel of the image')
# plt.show()

# from cv2 import inRange
# thresh_low = 100
# thresh_high = 255
# bw = inRange(img_r_blur, thresh_low, thresh_high)
# bw = bw ==255
# bw = bw.astype('uint8')
# masked = bw*img_r
# f,axes = plt.subplots(1,2,figsize=(10,4))
# axes[0].imshow(bw, cmap='gray')
# axes[1].imshow(masked, cmap ='gray')
# plt.show()

# h_masked = cv2.calcHist([img_r_blur], [0], bw, [256], [0,256])
# plt.plot(h_masked)
# plt.title("Histogram of the flower region not the background")
# plt.show()

# ret, thresh = cv2.threshold(img_r_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# mask = (thresh == 255).astype('uint8')
# h_masked_r = cv2.calcHist([img_r_blur], [0], bw, [256], [0,256])
# plt.plot(h_masked_r)
# plt.title("The threshold value is {}".format(ret))
# plt.show()
# plt.imshow(mask, cmap='gray')
# plt.show()

# # img_r = converted_img[:,:,0]  #red channel of the image
# # img_g = converted_img[:,:,1]  # Green channel of the image
# # img_b = converted_img[:,:,2]  # Blue channel of the image

# # h_r_object = cv2.calcHist([img_r], [0], mask, [256], [0,256])
# # h_g_object = cv2.calcHist([img_g], [0],mask, [256], [0,256])
# # h_b_object = cv2.calcHist([img_b], [0], mask, [256], [0,256])

# hist = []
# for i in range(3):
#     h = cv2.calcHist([converted_img[:,:,i]], [0], mask, [256], [0,256])
#     hist.append(h)
# hist = np.concatenate(hist, axis=0)
# plt.plot(hist)
# plt.title("Concatated Histogram of the object in the image")
# plt.show()

# h = [cv2.calcHist([converted_img], [i], mask, [256], [0,256]) for i in range(3)]
# f, axes = plt.subplots(1,3, figsize=(10,5))
# h - np.concatenate(h, axis=0)

# def main():
#     dataset_path = r'D:\SHU\ML_lab\Lab5_img_color_featuring_with_mask\flower_dataset'
#     image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith('.jpg')]
#     disply_original_grid(image_paths)
#     display_GaussianBlur_grid(image_paths)

# main()   


def main():
    dataset_path = r"D:\SHU\ML_lab\Lab5_img_color_featuring_with_mask\flower_dataset"
    image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(".jpg")]

    # Load images once (optimization)
    images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in image_paths]

    # Display original images
    display_grid(images)

    # Apply GaussianBlur to the red channel
    img_r = [img[:, :, 0] for img in images]
    img_r_blur = [cv2.GaussianBlur(img, (5, 5), 0) for img in img_r]
    display_grid(img_r_blur, cmap="gray")

    # Calculate histogram for Gaussian-blurred images
    h_r_blur = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in img_r_blur]

    # Apply thresholding
    thresholded_images, masked_images = apply_threshold(list(img_r_blur))

    # Display one example result
    display_grid(thresholded_images)
    display_grid(masked_images,)

    # Compute histograms with masks
    h_masked = [cv2.calcHist([img], [0], bw.astype(np.uint8), [256], [0, 256]) for img, bw in zip(img_r_blur, thresholded_images)]
    display_OTSU_histogram(h_masked)

    # Compute Otsu's thresholding
    _, thresh_images = zip(*[cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) for img in img_r_blur])

    # Compute histograms of thresholded images
    h_masked = []
    for index, (bw, img) in enumerate(zip(thresh_images, img_r_blur)):
        mask = (bw == 255).astype(np.uint8)
        h_masked_r = cv2.calcHist([img], [0], mask, [256], [0, 256])
        h_masked.append(h_masked_r)

    display_OTSU_histogram(h_masked)

main()