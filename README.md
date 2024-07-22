# Sobel-Edge-Detection
Implemented Sobel Edge Detection using CUDA and OpenCV in a high-performance computing environment in linux.

This program applies the Sobel Edge Detection algorithm using NVIDIA's CUDA in C++ to accelerate processing. Sobel Edge Detection which identifies edges/boundaries in an image by highlighting areas of rapid changes in high constrastThis algorithm is commonly used in image processing and computer vision. Using the algorithm I search the image for edges by calculating gradients in the x and in the y directions.  
The program can work with different image files such as '.jpg' file but can also work with other image files by specifying it. To Find the image in your system, make sure to set the file path correctly otherwise there will be errors with finding the image. 

## Features
- Edge Detection: Applying Sobel edge detection to images.
- Customizatino of image to suit different images and requirements.
- Input/Output: Loads image from system or upload to HPC environment, saves image and processes to new image file.

##Prerequisites
- I used C++11, but this program can be used with other versions of C++ if one were to revise the code
- NVIDIA's CUDA for GPU acceleration
- OpenCV for image processing and for converting from color to grayscale using 

The output after
![image](https://github.com/user-attachments/assets/2b114bd7-b604-4e7d-9e92-7bd05506fbe3)

##Contributing
Contributions are welcome, if you would like to contribute to the project please complete the following steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to the branch
5. Create a new Pull Request
