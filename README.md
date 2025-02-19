# Color Space and Histogram

Color space and histogram are fundamental concepts in computer vision that play crucial roles in image processing and analysis. These tools enable machines to interpret and manipulate visual information effectively.

**Color Space**

A color space is a specific organization of colors that allows for consistent representations across devices. Common color spaces include:

- RGB (Red, Green, Blue): Used in digital displays
- HSV (Hue, Saturation, Value): Intuitive for human perception
- CMYK (Cyan, Magenta, Yellow, Key/Black): Used in printing

In computer vision, different cooler spaces can highlight various aspects of an image, making certain operations more efficient or effective.

**Histogram**

A histogram is a graphical representation of the distribution of pixel intensities in an image. In the context of color images, histograms can be created for each color channel separately or as a combined representation.

Key uses of histograms in computer vision include:

- Image enhancement: Adjusting contrast and brightness
- Thresholding: Separating objects from backgrounds
- Feature extraction: Identifying unique characteristics of images

**Applications in Computer Vision**

The combination of color space manipulation and histogram analysis enables various computer vision tasks:

- Object detection and recognition
- Image segmentation
- Color-based tracking
- Image retrieval

By leveraging these tools, computer vision systems can process and understand visual data more effectively, leading to advancements in fields such as autonomous vehicles, medical imaging, and facial recognition.

 ## Task 1: RGB Color Space

Starting this week, we recommend using Matplotlib to display plots and images, as it offers more customizable functions. Today, you'll see a demonstration on how to use customization to generate plots for information visualization.

✅ Read a sample flower image by using `cv2.imread` function



✅ Check the data format of the image, what size is it?
>A color image is represented as a 3D matrix because it contains three color channels: Red, Green, and Blue (RGB). Each channel represents the intensity of that particular color for each pixel in the image:
>
>- Width: Represents the horizontal dimension of the image
>- Height: Represents the vertical dimension of the image
>- Depth: Represents the three color channels (R, G, B)
>
>So, for each pixel in the image, there are three values corresponding to the intensity of red, green, and blue. This three-dimensional structure allows for the representation of a wide range of colors by combining different intensities of these primary colors.
>
> ❗ In computer vision, understanding this 3D matrix structure is crucial for various tasks such as image processing, color-based object detection, and image segmentation.

✅ OpenCV have a different order for color channel, instead of R,G,B, they use B,G,R. which sometimes cause issues with other libraries (such as Matplotlib). To fix it, you can convert the color space from BGR to RGB by using `cv2.cvtColor()` function [Click here for more details](https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/)

## Task 2: After changing the channel order from BGR to RGB, create three separate images to store the R, G, and B channel images using list indexing and slicing.



## Task 3: RGB Histogram

✅ Use `cv2.calcHist()` to generate histogram for each image (R, G and B) [Click here for more details](https://www.geeksforgeeks.org/python-opencv-cv2-calchist-method/)

✅ Create a 1 by 3 subplot for three histograms. You can use Matplotlib plot function for that [click here for more details](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

✅ You can customize the plots, do some self-study and create a suitable plot as image histogram

## Task 4: HSV Color Space and histogram

✅ Use `cv2.cvtColor()` to convert the color space from RGB to HSV.

✅ (Optional) Create your own RGB2HSV function from scratch based on the lecture content.

✅ Create three separate channels (H, S, and V) for the image, similar to the RGB image.

✅ Display the channel images in a 1 by 3 subplot, using suitable color maps to represent the information.

✅ Display histograms of each channel in a 1 by 3 subplot

## Task 5: Compare the similarity of histograms

### Why Compare Histograms?

- Image Similarity: To determine how similar two images are in terms of color distribution.
- Object Recognition: To identify objects based on their color characteristics.
- Image Retrieval: To find images with similar content in large databases.
- Tracking: To track objects across video frames by comparing color histograms.

### How to Compare Histograms

There are several methods to compare histograms:

- **Correlation:** Measures the correlation between two histograms. Range: [-1, 1], where 1 is a perfect match.
- **Chi-Square:** Calculates the chi-square distance between histograms. Range: [0, ∞), where 0 is a perfect match.
- **Bhattacharyya Distance:** Measures the similarity of two probability distributions. Range: [0, 1], where 0 is a perfect match.

In OpenCV, you can use the `cv2.compareHist()` function to compare histograms using these methods.

✅ Do a research on each methods to compare histograms

✅ Implement some of your research and compare above histograms. For example, how similar between R and G channel?

> This is the warm up practice for next week's project - image segmentation.
