#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../ch3/common/book.h"

#ifndef N
#define N (1<<20) // 기본 벡터 길이 (약 백만). 빌드 시 -DN=... 로 변경 가능
#endif

#ifndef TPB
#define TPB 256    // block 당 thread 수
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 256 // grid 내 block 수 (65535 이하 권장). 빌드 시 -DNUM_BLOCKS=... 로 변경 가능
#endif

// grid-stride loop 방식: 매우 긴 N 에 대해서도 일정한 그리드/블록 크기로 처리 가능
__global__ void add_kernel(const int *a, const int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  // 다음으로 건너뛸 간격

    while (tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += stride;
    }
}

int main() {
    std::printf("[vector_add_stride] N=%d, TPB=%d, NUM_BLOCKS=%d\n", N, TPB, NUM_BLOCKS);

    int *h_a = (int*)std::malloc(sizeof(int) * N);
    int *h_b = (int*)std::malloc(sizeof(int) * N);
    int *h_c = (int*)std::malloc(sizeof(int) * N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 1; // 더하기가 쉬운 값으로 설정
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_c, sizeof(int) * N));

    HANDLE_ERROR(cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice));

    dim3 threads(TPB);
    dim3 blocks(NUM_BLOCKS);

    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    SYNC_AND_CHECK();

    HANDLE_ERROR(cudaMemcpy(h_c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    // 결과 검증(샘플)
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) { ok = false; break; }
    }
    std::printf("result: %s\n", ok ? "OK" : "NG");

    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    return 0;
} 