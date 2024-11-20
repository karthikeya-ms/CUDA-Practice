#include <iostream>
#include <cuda_runtime.h>

__global__ void parallel_reduction(const float *d_a, float *sum, int n){

    __shared__ float partialSum[256];

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int threadIdxInBlock = threadIdx.x;

    partialSum[threadIdxInBlock] = (idx<n) ? d_a[idx] : 0.0f;

    __syncthreads();

    //Reduce within block
    for(int stride = blockDim.x/2 ; stride > 0 ; stride /=2){

        if(threadIdxInBlock < stride){
            partialSum[threadIdxInBlock] += partialSum[threadIdxInBlock + stride];
        }
        __syncthreads();
    }

    // write the blocks partial sum to global memory
    if (threadIdxInBlock ==0){

        atomicAdd(sum, partialSum[0]);
    }

}



int main(){

    int n = 1000;
    size_t bytes = n*sizeof(float);

    float *h_a = new float[n];
    float h_sum = 0.0f;

    for(int i=0; i<n; i++){

        h_a[i] = i*1.0f;
    }  

    float *d_a, *d_sum;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize -1)/blockSize;
    parallel_reduction<<<gridSize, blockSize>>>(d_a, d_sum, n);

    cudaMemcpy(&h_sum, d_sum, sizeof(float),cudaMemcpyDeviceToHost);

   
    std::cout<<"Sum:"<<h_sum<<" ";
    std::cout<<std::endl;

    cudaFree(d_a);
    cudaFree(d_sum);

    delete[] h_a;

    return 0;
}