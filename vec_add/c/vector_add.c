#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define N 1000000000
#define MAX_ERR 1e-6

void vector_add(float *out, float *a, float *b, int n)
{
	for(int i = 0; i < n; i++)
	{
		out[i] = a[i] + b[i];
	}
}

int main()
{
	float *a, *b, *out;

	// allocate memory
	a = (float*)malloc(sizeof(float)*N);
	b = (float*)malloc(sizeof(float)*N);
	out = (float*)malloc(sizeof(float)*N);

	// initialize array
	for(int i = 0; i < N; i++)
	{
		a[i] = 1.0f; 
		b[i] = 2.0f;
	}

	// execute func
	clock_t t;
	t = clock();
	vector_add(out, a, b, N);
	t = clock() - t;

	// verification
	for(int i = 0; i < N; i++)
	{
		assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
	}
	printf("PASSED\n");
	

	// dealloc memory
	free(a);
	free(b);
	free(out);

	double time_taken = (((double)t)/CLOCKS_PER_SEC)*1000;
	printf("time taken: %f ms\n", time_taken);

	return 0;
}
