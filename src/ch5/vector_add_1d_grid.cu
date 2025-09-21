#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../ch3/common/book.h"

#ifndef N
#define N 1000  // 기본 벡터 길이 (빌드 시 -DN=... 로 변경 가능)
#endif

#ifndef TPB
#define TPB 128 // block 당 thread 수 (Threads Per Block)
#endif

// 글로벌 스레드 인덱싱: threadIdx.x + blockIdx.x * blockDim.x
__global__ void add_kernel(const int *a, const int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    std::printf("[vector_add_1d_grid] N=%d, TPB=%d\n", N, TPB);

    int *h_a = (int*)std::malloc(sizeof(int) * N);
    int *h_b = (int*)std::malloc(sizeof(int) * N);
    int *h_c = (int*)std::malloc(sizeof(int) * N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 3;
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_c, sizeof(int) * N));

    HANDLE_ERROR(cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice));

    dim3 threads(TPB);
    dim3 blocks((N + TPB - 1) / TPB);  // 총 스레드 수가 N 이상이 되도록 계산

    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    SYNC_AND_CHECK();

    HANDLE_ERROR(cudaMemcpy(h_c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    // 결과 스팟 체크
    for (int i = 0; i < 10 && i < N; ++i) {
        std::printf("c[%d] = %d\n", i, h_c[i]);
    }

    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    return 0;
} 