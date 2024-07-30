# Sobel-Edge-Detection
Implemented Sobel Edge Detection using CUDA and OpenCV in a high-performance computing environment in linux.

This program applies the Sobel Edge Detection algorithm using NVIDIA's CUDA in C++ to accelerate processing. It was origanally assigned to me from Professor David Kaeli to learn what Sobel Edge Detection is then take any image I want and perform edge detection on it, to detect any significant changes in the image.

First I researched what Sobel was, Sobel Edge Detection is an algorithm which identifies edges/boundaries in an image 
by highlighting areas of rapid changes in high constrast. This algorithm is commonly used in image processing and computer vision. Using the algorithm I search the image for edges by calculating gradients in the x and in the y directions.  
The program can work with different image files such as '.jpg' file but can also work with other image files by specifying it. To Find the image in your system, make sure to set the file path correctly otherwise there will be errors with finding the image. 

## Design an Implementation
To achieve the detection, I plan to:
- First take the image and convert it from a RGB to a Gray-Scale format. This can be done using OpenCV to load the image and convert it to the grayscale.
- Once that is completed do a use an invariant operation like thresholding to create a binary image for any dramatic changes or blurring.
- Perform the Edge Detection using a Sobel function and CUDA (this can be looked into based of how other people have implemented Sobel in C or Python)
  - Make sure to use the correct version of CUDA based off the GPU that will be given. Also check the version of OpenCV to use as well. 
- Combine both results by taking the gradients
- use OpenCV to dave the image to the directory
- Open the image to verify that the conversion was done correctly and then output the dimensions that had the detection done to them.
    - Figure out the command that allows you to open an image in Linux 
## Features
- Edge Detection: Applying Sobel edge detection to images.
- Grayscale Conversion: Using OpenCV's *COLOR_RGB2GRAY* to convert the colored image to gray. 
- Customizatino of image to suit different images and requirements.
- Input/Output: Loads image from system or upload to HPC environment, saves image and processes to new image file.
- Gausian Blurring: reduces noise before edge detection

## Prerequisites
- I used C++11, but this program can be used with other versions of C++ if one were to revise the code
- NVIDIA's CUDA for GPU acceleration, specificaly CUDA 11.4
- OpenCV for image processing and for converting from color to grayscale using 

To compile the code, ensure you have a compatible 'nvcc' and C++ compiler and GPU installed. an example build command is `nvcc -arch=sm_35 file_name.cu -o file_executable $(pkg-config --cflags --libs opencv)`. Then do the normal execution method in Linux. 
Be sure to adjust the command according to your CUDA, OpenCV, and GPU setup.

## Results
Input image before applying Sobel:
![test_photo](https://github.com/user-attachments/assets/09ca2852-1290-4a29-a1b6-caf135e35421)

Output image after applying Sobel:
![Screenshot 2024-07-22 162056](https://github.com/user-attachments/assets/632cd295-923f-4cab-a0f2-fcaa18606eec)

The dimensions of the image to ensure to ensure that the image was being processed succesfully: 

Image has been loaded 

Dimensions: 1127, 683 

Number of channels: 3 

Image has been converted. 

Dimesnsion of gray version: 1127, 683

Number of grey channels: 1 

Blurred image dimensions: 1127 x 683 

Thresholded image dimensions: 1127 x 683 

Edge detection has been performed and saved to another file. 

## Author
Stephen Sodipo

## Contributing
Contributions are welcome, if you would like to contribute to the project please complete the following steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to the branch
5. Create a new Pull Request

## Resources Used: 
- Image Processing Learnging Resources page on what the [Sobel Edge Detector](https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm) is.
- When researching what Sobel was I came across this article on what Sobel was and how you could connect CUDA to it: [Implementing a Sobel Filter with CUDA in Python](https://medium.com/@deepika.vadlamudi/implementing-a-sobel-filter-with-cuda-in-python-2b9b18485e31).
- This Document gave me the inspiration to break the assignment into smaller steps and how to approach each step in a slower fashion, than my initial approach, [Comparison of Sobel in C, OpenCV, and CUDA](https://danyele.github.io/lecture_notes/SPD_Project_Report_Daniele_Gadler_5_0.pdf).
- Helped me figrue out how to compile the cuda program with opencv: [compiling opencv](https://stackoverflow.com/questions/9094941/compiling-opencv-in-c)
- Documentation of OpenCV on converting a colored image to the grayscale: [OpenCV Documentation](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ae50d0c66ee53e974234ac84cf51d1d4e)
- Another source I used to figure out converting the colors [converting colors using OpenCV](https://www.tutorialspoint.com/how-to-convert-color-spaces-in-opencv-using-cplusplus).
- Stumbled upon this persons repository on how they apprached using a thread to parallize the iamge. [Github Sobel Filter](https://github.com/lukas783/CUDA-Sobel-Filter/tree/master)
- This was another document that helped me understand Sobel and utilize CUDA to provide performance benefits, as it is parallize processing on the different pixels to speed up the computation compared to a CPU implementation [Implementation of Sobel filter using CUDA](https://iopscience.iop.org/article/10.1088/1757-899X/1045/1/012016/pdf).
