#include <iostream>
#include <cuda_runtime.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int n)
{
	std::cout << "hey" << "\n";
	int size = n * sizeof(float);
	float *A_d, *B_d, *C_d;

	cudaMalloc((void **)&A_d, size);
	cudaMalloc((void **)&B_d, size);
	cudaMalloc((void **)&C_d, size);

	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// addition here

	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
};

int main()
{
	float A_h{}, B_h{}, C_h{};

	vecAdd(&A_h, &B_h, &C_h, 1);
};
