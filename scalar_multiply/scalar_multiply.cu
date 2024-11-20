/*
Objective: Multiply every element of a vector with a scalar value

1) Allocate the memory in both host and device
2) Copy the input vector into device global memory
3) Call the kernel function
4) Copy the output from device to host
5) Free memory on host and device
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void scalar_multiply(float *d_a, float scalar, int n){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<n){
        d_a[idx] = scalar*d_a[idx];
    }
}

int main(){

    int n = 1000;
    size_t bytes = n*sizeof(float);

    float *h_a = new float[n];

    for(int i=0; i<n; i++){

        h_a[i] = i*1.0f;
    }

    //Allocate memory on device
    float *d_a;
    cudaMalloc(&d_a, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize -1 )/blockSize;

    scalar_multiply<<<gridSize, blockSize>>>(d_a, 3.0f, n);

    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++){

        std::cout<<h_a[i]<<" ";
    }
    std::cout<<std::endl;

    cudaFree(d_a);

    delete[] h_a;

    return 0;




}