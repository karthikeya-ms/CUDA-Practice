#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

inline float sinsum(float x, int N) {
    float term = x;
    float sum = term;
    float x2 = x * x;
    for (int n =1; n<N; n++) {
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
    double cpu_sum = 0.0;

    for(int step = 0; step < steps; step++) {
        float x = step * step_size;
        cpu_sum += sinsum(x, terms);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "CPU Time: " << duration << " microseconds" << endl;    

    //Trapezoidal rule correction
    cpu_sum -= (sinsum(0, terms) + sinsum(pi, terms)) * 0.5;
    cpu_sum *= step_size;
    // Output the result with no of steps and terms and stime taken
    cout << "Steps: " << steps << ", Terms: " << terms << ", Result: " << cpu_sum << endl;
    return 0;
}


