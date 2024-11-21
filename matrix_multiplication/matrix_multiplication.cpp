#include <iostream>

int main(){

    const int M = 10, K = 20, N = 15;
    size_t abytes = M*K*sizeof(float), bbytes = K*N*sizeof(float), cbytes = M*N*sizeof(float);

    float h_a[M*K], h_b[K*N], h_c[M*N];

    for(int i = 0; i<M*K; i++){
        h_a[i] = i*2.0f;
        std::cout<<h_a[i]<<" ";
    }

    

    return 0;

}
