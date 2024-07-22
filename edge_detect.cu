#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
using namespace cv;

// error checking for cuda calls
void checkCudaError(cudaError_t result, char const *const func, const char *const file, int const line) {
        if (result != cudaSuccess) {
        //so if the cuda doesn't compute all the way then it will close out and tell me why
                fprintf(stderr, "CUDA error at %s:%d code = %d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        //make it exit with the error
        exit(EXIT_FAILURE);
        }
}
//constant that is used to call the error checker method
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)

// Have error checking for the opencv here
void checkOPCVError(bool condition, const char* message) {
        if(!condition) {
                cerr << "OpenCV error: " << message << endl;
                exit(EXIT_FAILURE);
        }
}

// method to perform teh Sobel for the X axis
__global__ void sobelXKern(const unsigned char* d_in, float* d_out, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) {
                // perform teh sobel on the x level
                int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

                float gradX = 0;
                for(int ky = -1; ky <= 1; ++ky) {
                        for(int kx = -1; kx <= 1; ++kx) {
                                int ix = min(max(x + kx, 0), width - 1);
                                int iy = min(max(y + ky, 0), height - 1);
                                gradX += d_in[iy * width + ix] * sobelX[ky + 1][kx + 1];
                        }
                }
                d_out[y * width + x] = abs(gradX);
        }
}

// method to perform teh Sobel for the Y axis
__global__ void sobelYKern(const unsigned char* d_in, float* d_out, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < width && y < height) {
                // perform teh sobel on the x level
                int sobelY[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

                float gradY = 0;
                for(int ky = -1; ky <= 1; ++ky) {
                        for(int kx = -1; kx <= 1; ++kx) {
                                int ix = min(max(x + kx, 0), width - 1);
                                int iy = min(max(y + ky, 0), height - 1);
                                gradY += d_in[iy * width + ix] * sobelY[ky + 1][kx + 1];
                        }
                }
                d_out[y * width + x] = abs(gradY);
        }
}

// main
int main() {
        // get the path to the image
        const char* filename = "/scratch/sodipo.s/workspace/testimage.jpg";

        // load the image
        Mat img = imread(filename);
        checkOPCVError(!img.empty(), "Unable to load image!");

        // get dimensions of the image (maybe in pixels)
        cout << "Image has been loaded" << endl;
        cout << "Dimensions: " << img.cols << ", " << img.rows << endl;
        cout << "Number of channels: " << img.channels() << endl;

        int width = img.cols;
        int height = img.rows;
        int imgSize = width * height * sizeof(unsigned char);
        int gradSize = width * img.rows * sizeof(float);

        // convert the image to grayscale
        Mat grayImg;
        cvtColor(img, grayImg, COLOR_BGR2GRAY); // this will take the image and make it grayscale

        // print the grayscale so i can check
        cout << "Image has been converted." << endl;
        // make sure the dimensions have not changed
        cout << "Dimesnsion of gray version: " << grayImg.cols << ", " << grayImg.rows << endl;
        cout << "Number of grey channels: " << grayImg.channels() << endl;

        // have a Gaussian blur to the image to reduce nose
        Mat blurredImg;
        GaussianBlur(grayImg, blurredImg, Size(5, 5), 0);

        // appy a threshold to see if there are any changes ni teh size
        Mat thresHoldImg;
        double thresHoldval = 256;
        threshold(blurredImg, thresHoldImg, thresHoldval, 255, THRESH_BINARY);

        // see the properties
        cout << "Blurred image dimensions: " << blurredImg.cols << " x " << blurredImg.rows << endl;
        cout << "Thresholded image dimensions: " << thresHoldImg.cols << " x " << thresHoldImg.rows << endl;

        //Allocate memory on the GPu
        unsigned char *d_in;
        float *d_outX, *d_outY;

        CHECK_CUDA_ERROR(cudaMalloc(&d_in, imgSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_outX, gradSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_outY, gradSize));

        // copy image data
        CHECK_CUDA_ERROR(cudaMemcpy(d_in, grayImg.data, imgSize, cudaMemcpyHostToDevice));

        //define block and grid size
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // call the kernel
        sobelXKern<<<gridSize, blockSize>>>(d_in, d_outX, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());
        sobelYKern<<<gridSize, blockSize>>>(d_in, d_outY, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // allocate memory for the cpu to save the results
        float *h_outX = new float[width * height];
        float *h_outY = new float[width * height];
        Mat result(height, width, CV_32FC1);

        // Copy the results
        CHECK_CUDA_ERROR(cudaMemcpy(h_outX, d_outX, gradSize, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_outY, d_outY, gradSize, cudaMemcpyDeviceToHost));

        // combine teh sobel results
        for(int i = 0; i < width * height; ++i) {
          result.at<float>(i / width, i % width) = sqrt(h_outX[i] * h_outX[i] * h_outY[i]);
        }

        // save the result to a new file
        imwrite("sobel_edge_detected.jpg", result);

        // Free GPU memory
        CHECK_CUDA_ERROR(cudaFree(d_in));
        CHECK_CUDA_ERROR(cudaFree(d_outX));
        CHECK_CUDA_ERROR(cudaFree(d_outY));

        delete[] h_outX;
        delete[] h_outY;

        cout << "Edge detection ahs been performed and saved to another file." << endl;

        return 0;
}
