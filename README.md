# Image Manipulation

To apply what we have learned in the lecture, you are required to complete this task sheet. You can seek assistance from tutors, fellow students, and/or online resources.

Please follow the instructions. In order to make your code work, you need to complete all the tasks in the checklist - ✅


## Task 1: Use OpenCV to load and display an image

OpenCV (Open Source Computer Vision Library: http://opencv.org) is an open-source library that includes several hundreds of computer vision algorithms.

In the lab, we will use the OpenCV library to work on image processing and computer vision tasks. Let's start from the basics

First of all, you need to install OpenCV library.

✅ you  need to `import` it into your code. the library is called `cv2`

To load and display image, you need to:
> ✅ Choose some sample images and put them in your Google Drive
>
> ✅ Allow this project to visit your Google Drive
>
> ✅ Use `cv2.imread()` function. You can get the help document from [Here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
> ✅ Use `cv2_imshow()` function to display the image

❗The OpenCV document is originally designed for C++, the Python doc is include but is below the C++ doc

## Task 2: Basic Image Manipulations

__1. Resize the image__

✅ Check the variable you created to store the image, what data format/size is it? You can `import` library called `numpy` to return the size of array by using `numpy.shape()`

✅ Use `cv2.resize()` function to change the image size to 100 by 100. Display the new image.

__2.Rotate the image 45 degrees__

✅ Creates a rotation matrix using `cv2.getRotationMatrix2D()`. This function takes three arguments: the center of the rotation, the angle of rotation, and the scale.

✅ Rotates the image using cv2.warpAffine(). This function takes three arguments: the image, the rotation matrix, and the size of the output image.

❗How to get the center of the rotation?

❗Be aware the data format of the center, which should be a tuple as `(x, y)`
> A tuple is a data structure in Python that is similar to a list. However, there are some key differences.
Tuples are immutable, which means that they cannot be changed after they are created. Lists, on the other hand, are mutable, which means that they can be changed.
Tuples are ordered, which means that the elements in a tuple have a specific order.
Tuples can contain elements of different data types.

✅ Display the rotated image

__3. Applying Filters (Convolution)__

You can apply many filters to the images, such as Gaussian Blur filters to blur the image

✅ Using `cv2.GaussianBlur()` to blur an image and display the output

`cv2.GaussianBlur()` is a function that applies a Gaussian blur to an image.
* The first argument is the image to be blurred.
* The second argument is the kernel size. you can use for example (3, 3),(5, 5), (9, 9).
* The third argument is the standard deviation of the Gaussian distribution. In this case, it is 0, which means that the standard deviation is calculated automatically.

✅ There are more filters in OpenCV. Such as Median filter (`cv2.medianBlur()`) which is a highly effective filter for reducing noise but also keep the share edge of the image (the Gaussian blurs everything including the sharp edges). Work on yourself and try different cv2 function and display their results.

## Task 3: Save the image

Once you completed the image processing, if you want to save the image back to computer as an image, you will use `cv2.imwrite()`. Check out how to use them and save your image from the task 2 in your Google Drive.

## Task 4 - Manipulate pixels

If you get the size and the data structure of the variable used to store the image, you then can use the __indexing method__ to visit its elements - pixels.
Check this [doc](https://numpy.org/devdocs/user/basics.indexing.html) for the indexing technique

> ✅ Display the pixel values at (x, y) location (20,20).
>
> ✅ Be careful if your image is a color image and you need to choose an image channel (i.e. Blue, Green or Red channel). Can you display just the value from __Red__ channel?
>
> ✅__[Bonus]__Check the given soucecode of the function called `salt_pepper()`. Try to make it work on your own image. What does this function do?
> ✅__[Bonus]__The image generated is noisy, you can use the filters we used in the previous tasks to reduce the noise. Which filter works better?

# Task 5: Feature Space [Bonus Task]

✅ Load the 20 images of cats and dogs from the "CAT_VS_DOG" image folder. Find the [details](https://www.geeksforgeeks.org/how-to-use-glob-function-to-find-files-recursively-in-python/) on how to use `glob` to load image batches.

✅ Reshape all the images to a uniform size of 200*200 pixels. You may want to check the [examples](https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0) of how to resize an image in OpenCV.

✅ Design a feature space. OpenCV have some prebuild features such as `cv2.goodFeaturesToTrack()` There are some other features you can play with such as:
* SIFT (Scale-Invariant Feature Transform)
* SURF (Speeded-Up Robust Features)
* ORB (Oriented FAST and Rotated BRIEF)

Each feature has their advantages for a specific computer vision task. (btw, We will introduce HoG feature in the classroom for object detection tasks). Try to do a research and self-study to see how the information can be modelled in Python as a feature space. As we mentioned in the lecture, each image can be represented by a row of features. The whole dataset are then formatted together as a 2D datasheet. I

✅ Implement your design and generate a comprehensive feature space in Python for this dataset. You have the flexibility to use various data structures such as lists or arrays. The feature space should encompass all the image samples and incorporate all the features you have designed for this feature space.

ℹ You may want to check how to use [list comprehansion](https://www.w3schools.com/python/python_lists_comprehension.asp) rather than for loops to process a group of data, which is much faster!
