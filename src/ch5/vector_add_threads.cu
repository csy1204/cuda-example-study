#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../ch3/common/book.h"  // 에러 체크 매크로

#ifndef N
#define N 16  // 기본 벡터 길이 (빌드 시 -DN=... 로 변경 가능)
#endif

// GPU에서 실행되는 커널 함수
// - 이 예제는 "block 1개, thread N개" 구성입니다: add<<<1, N>>>
// - 각 thread는 자신만의 인덱스 threadIdx.x 를 사용하여 한 원소를 담당합니다.
__global__ void add_kernel(const int *a, const int *b, int *c, int n) {
    int tid = threadIdx.x;  // 블록 내부에서의 스레드 인덱스 (0 ~ blockDim.x-1)
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    std::printf("[vector_add_threads] N=%d\n", N);

    // 1) 호스트(CPU) 메모리 준비
    int *h_a = (int*)std::malloc(sizeof(int) * N);
    int *h_b = (int*)std::malloc(sizeof(int) * N);
    int *h_c = (int*)std::malloc(sizeof(int) * N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;          // 0,1,2,3,...
        h_b[i] = i * 2;      // 0,2,4,6,...
    }

    // 2) 디바이스(GPU) 메모리 할당
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(int) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_c, sizeof(int) * N));

    // 3) 입력 데이터를 GPU로 복사
    HANDLE_ERROR(cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice));

    // 4) 커널 실행: 블록 1개, 스레드 N개
    add_kernel<<<1, N>>>(d_a, d_b, d_c, N);
    SYNC_AND_CHECK();  // 커널 완료 및 에러 체크

    // 5) 결과를 호스트로 복사
    HANDLE_ERROR(cudaMemcpy(h_c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

    // 6) 결과 확인 (앞 몇 개만)
    for (int i = 0; i < N; ++i) {
        std::printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // 7) 자원 해제
    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    return 0;
} 