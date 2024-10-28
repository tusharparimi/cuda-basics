#include <iostream>
#include <opencv2/opencv.hpp>
#include <typeinfo>


__global__ void rgb2gray_kernel(unsigned char *vec, unsigned char *out_vec, int n)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n)
    {
        int r = vec[tid * 3];
        int g = vec[tid * 3 + 1];
        int b = vec[tid * 3 + 2];
        out_vec[tid] = static_cast<unsigned char>(0.2989*r + 0.5870*g + 0.1140*b);
    }

}


int main()
{
    cv::Mat og_image;
    og_image = cv::imread("../../../puppy.jpg");
    cv::Mat image;
    // cv::resize(og_image, image, cv::Size(600, 400));
    image = og_image;
    cv::imshow("image", image);
    cv::waitKey(0);

    int N = image.size[0] * image.size[1] * image.channels();
    int n = image.size[0] * image.size[1];
    std::cout << "N: " << N << ", n: " << n << std::endl;

    cv::Mat flat_image = image.reshape(1, 1);

    std::cout << "original image type(" << cv::typeToString(image.type()) << ") :";
    std::cout << image.channels() << " x " << image.size << std::endl;
    std::cout << "reshaped image type(" << cv::typeToString(flat_image.type()) << ") :";
    std::cout << flat_image.channels() << " x " << flat_image.size << std::endl;

    unsigned char *image_array, *out;
    unsigned char *d_image_array, *d_out;

    image_array = (unsigned char*)malloc(sizeof(unsigned char) * N);
    out = (unsigned char*)malloc(sizeof(unsigned char) * n);


    for(int i = 0; i < N; i++)
    {
        image_array[i] = flat_image.at<unsigned char>(0, i);
    }

    cudaMalloc((void**)&d_image_array, sizeof(unsigned char) * N);
    cudaMalloc((void**)&d_out, sizeof(unsigned char) * n);
    
    cudaMemcpy(d_image_array, image_array, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);
    
    //execute kernel
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1)/threads_per_block;
    rgb2gray_kernel<<< blocks_per_grid , threads_per_block >>>(d_image_array, d_out, n);

    cudaMemcpy(out, d_out, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost);

    cv::Mat out_image = cv::Mat(image.size[0], image.size[1], CV_8UC1, out);

    std::cout << "out image type(" << cv::typeToString(out_image.type()) << ") :";
    std::cout << out_image.channels() << " x " << out_image.size << std::endl;

    cv::imshow("grayscale", out_image);
    cv::waitKey(0);

    cudaFree(d_out);
    cudaFree(d_image_array);

    free(image_array);
    free(out);

    return 0;
}
