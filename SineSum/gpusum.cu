#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <thrust/device_vector.h>
using namespace std;

__host__ __device__ inline float sinsum(float x, int terms){
    float x2 = x*x;
    float term = x;
    float sum = term;

    for (int n =1; n< terms; n++){
        term *= -x2 /(2*n*(2*n+1));
        sum += term;
    }
    return sum;
}   

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){
    int step = blockIdx.x*blockDim.x + threadIdx.x;
    if (step < steps){
        float x = step * step_size;
        sums[step] = sinsum(x, terms);
    }
}

int main(int argc, char* argv[]){
    int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;

    double pi = 3.14159265358979323846;
    double step_size = pi / (steps-1);

    int threads = 256;
    int blocks = (steps + threads - 1) / threads;

    //Allocate memory on the GPU and get a pointer to it
    thrust::device_vector<float> dsums(steps); 
    float *dptr = thrust::raw_pointer_cast(&dsums[0]);

    auto start = chrono::high_resolution_clock::now();
    gpu_sin<<<blocks, threads>>>(dptr, steps, terms, step_size);
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "GPU Time: " << duration << " microseconds" << endl;
    gpu_sum -= (sinsum(0, terms) + sinsum(pi, terms)) * 0.5;
    gpu_sum *= step_size;
    cout << "GPU Sum: " << gpu_sum << endl;
    return 0;
}



