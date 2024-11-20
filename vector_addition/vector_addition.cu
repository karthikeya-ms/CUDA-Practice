/* Problem: Vector addition

1) Allocate memory for vectors on both host and device
2) Copy the input vectors from host to device memory
3) Launch a CUDA kernel to perform the addition on GPU
4) Copy the result vector back to the host
5) Free device memory

*/

#include <iostream>
#include <cuda_runtime.h>


__global__ void vector_add(const float* a, const float* b, float* c, int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n){
        c[idx] = a[idx] + b [idx];
    }
}

int main() {

    int n = 1000; //vector size
    size_t bytes = n*sizeof(float);

    //Allocate memory on host
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    //Initialize input vectors
    for(int i =0; i<n; i++){

        h_a[i] = i*1.0f; //single precision floating point numbers as GPUs are optimized for those - requires less memory and computation
        h_b[i] = i*2.0f;
    }

    //Allocate memory on device
    float *d_a, *d_b, *d_c; //pointers to objects stored in device global memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);


    //Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


    //Launch kernel (1d grid and block config)

    int blockSize = 256;
    int gridSize = (n + blockSize -1)/blockSize;
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    //Copy result back to host

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //Print a few results
    for(int i = 0; i<10; i++){

        std::cout << h_c[i] << " ";
    }

    std::cout << std::endl;

    //Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    //Free host memory
    delete[] h_a;   
    delete[] h_b;
    delete[] h_c;

    return 0;

}