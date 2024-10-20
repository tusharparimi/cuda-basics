#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define N 10000000
#define MAX_ERR 1e-6

//  kernel func
__global__ void vector_add(float *out, float *a, float *b, int n)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n)
	{
		out[tid] = a[tid] + b[tid];
	}
}

int main()
{
	float *a, *b, *out;
	float *d_a, *d_b, *d_out;

	// alloc host memory
	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	out = (float*)malloc(sizeof(float) * N);

	// initialize array
	for(int i = 0; i < N; i++)
	{
		a[i] = 1.0f;
		b[i] = 2.0f;
	}

	// alloc device memory
	cudaMalloc((void**)&d_a, sizeof(float) * N);
	cudaMalloc((void**)&d_b, sizeof(float) * N);
	cudaMalloc((void**)&d_out, sizeof(float) * N);

	// copy a & b to device
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

	int block_size = 256;
	int grid_size = (N + block_size)/block_size;
	// execute kernel
	vector_add<<< grid_size , block_size >>>(d_out, d_a, d_b, N);

	// copy result back to host memory
	cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

	// verification
	for(int i = 0; i < N; i++)
	{
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("PASSED\n");

	// deallocate device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_out);

	// deallocate host memory
	free(a);
	free(b);
	free(out);

	return 0;
}
