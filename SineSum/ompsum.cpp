#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
using namespace std;

float sinsum(float x, int terms) {
    float term = x;
    float sum = term;
    float x2 = x * x;
    for (int n = 1; n < terms; n++) {
        term *= -x2 / (float)((2*n)*(2*n+1));
        sum += term;
    }
    return sum;
}

int main(int argc, char* argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;

    double pi = 3.14159265358979323846;
    double step_size = pi / (steps-1);

    auto start = chrono::high_resolution_clock::now();
    double omp_sum = 0.0;   
    omp_set_num_threads(8);
    #pragma omp parallel for reduction(+:omp_sum)
    for (int step = 0; step < steps; step++) {
        float x = step * step_size;
        omp_sum += sinsum(x, terms);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "CPU Time: " << duration << " microseconds" << endl;
    omp_sum -= (sinsum(0, terms) + sinsum(pi, terms)) * 0.5;
    omp_sum *= step_size;
    cout << "CPU Sum: " << omp_sum << endl;
    return 0;
}

// g++ -fopenmp -o omp_sum omp_sum.cpp
// ./omp_sum 1000000 1000
// obtained a speedup up of 5x omn Intel(R) Core(TM) i5-8250U CPU (4 cores and 8 threds)
