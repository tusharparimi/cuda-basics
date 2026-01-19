#include <cuda_runtime.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>


__global__  void vecAdd(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex]; 
    }
}

void initArray(float* A, int length) {
    std::srand(std::time({}));
    for (int i=0; i < length; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C, int length) {
    for (int i=0; i < length; i++) {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001) {
    for (int i=0; i<length; i++) {
        if (fabs(A[i] - B[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void unifiedMemExample(int vectorLength) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    cudaDeviceSynchronize();

    serialVecAdd(A, B, comparisonResult, vectorLength);

    if (vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
        printf("Unified Memeory: CPU and GPU answers match\n");
    }
    else {
        printf("UnifiedMemory: Error - CPU and GPU answers do not match\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}

int main(int argc, char* argv[]) {
    int vectorLength = 1024;
    if (argc >= 2) {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}