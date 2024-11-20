/* Objective: Compute the dot product of two vectors using CUDA

1) Allocate memory for vectors on both host and device
2) Copy the input vectors from host to device memory
3) Launch a CUDA kernel to perform the addition on GPU
4) Copy the result vector back to the host
5) Free device memory

a) Each thread computes the dot product of corresponding elements
b) Atomic operations to sum these products in device memory 

*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void dot_product(const float* a, const float*b, float*c, float*d_sum, int n){

    __shared__ float partialSum[256]; //shared memory for partial sums

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int threadIdxInBlock = threadIdx.x;

    //Initialize shared memory
    partialSum[threadIdxInBlock] = (idx<n) ? a[idx]*b[idx] : 0.0f;

    __syncthreads();

    //Reduce within block
    for(int stride = blockDim.x/2 ; stride >0 ; stride /=2){

        if(threadIdxInBlock < stride){
            partialSum[threadIdxInBlock] += partialSum[threadIdxInBlock + stride];
        }
        __syncthreads();
    }

    // write the blocks partial sum to global memory
    if (threadIdxInBlock ==0){

        atomicAdd(d_sum, partialSum[0]);
    }


    if(idx<n){
        c[idx] = a[idx]*b[idx];
    }

}

int main() {

    int n = 1000;
    size_t bytes = n*sizeof(float);



    //Allocate host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    float h_sum = 0.0f;

    //Initialize vectors h_a and h_b
    for(int i=0; i<n; i++){

        h_a[i] = i*1.0f;
        h_b[i] = i*2.0f;
    }


    //Allocate memory on device
    float *d_a, *d_b, *d_c, *d_sum;
     

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_sum, sizeof(float));


    //Copy the vectors h_a and h_b to device global memory
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum,sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize -1)/blockSize;
    dot_product<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_sum, n);

    // Copy the result vector and dot product to host memory
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    //Print a few results
    for(int i=0; i<10; i++){

        std::cout<<h_c[i]<<" ";
        
    }
    std::cout<<std::endl; 

    std::cout<<"Dot product of vectors h_a and h_b is:"<<h_sum;

    //Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_sum);

    //Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;


    return 0;

}