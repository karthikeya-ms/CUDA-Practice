/*
Objective: Rectangular matrix multiplication
*/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matrix_multiplication(const float *d_a, const float *d_b, float *d_c, int M, int K, int N){

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float value = 0.0f;

        for(int k = 0; k < K; k++){
            value += d_a[row*K + k]*d_b[k*N + col];
        }
        d_c[row*N + col] = value;
    }   
}

int main(){

    const int M = 10, K = 20, N = 15;
    size_t abytes = M*K*sizeof(float), bbytes = K*N*sizeof(float), cbytes = M*N*sizeof(float);

    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N);

    for(int i = 0; i<M*K; i++){
        h_a[i] = 1*1.0f;
    }

    for(int i = 0; i<K*N; i++){
        h_b[i] = 1*1.0f;
    }

    //Allocate memory on device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, abytes);
    cudaMalloc(&d_b, bbytes);
    cudaMalloc(&d_c, cbytes);

    //Copy matrices to device

    cudaMemcpy(d_a, h_a.data(), abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bbytes, cudaMemcpyHostToDevice);

    dim3 blockDim(10,10);
    dim3 gridDim((N+blockDim.x - 1)/blockDim.x, (M+blockDim.y - 1)/blockDim.y);

    matrix_multiplication<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);

    cudaMemcpy(h_c.data(), d_c, cbytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout<<std::endl;
    }

    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    

}

