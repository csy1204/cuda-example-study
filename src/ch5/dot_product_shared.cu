#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../ch3/common/book.h"

#ifndef N
#define N (1<<20) // 요소 수 (약 백만)
#endif

#ifndef TPB
#define TPB 256    // block 당 thread 수 (2의 거듭제곱 권장)
#endif

// 공유 메모리를 이용한 블록 단위 reduction
// - 각 thread 가 부분 곱(a[i]*b[i])을 계산해 __shared__ 배열에 저장
// - __syncthreads() 로 동기화 후, 블록 내에서 합산(reduction)을 수행
// - 블록의 최종 합을 전역 메모리의 partial_sums[blockIdx.x] 에 저장
__global__ void dot_block_reduce_kernel(const float *a, const float *b, float *partial_sums, int n) {
    __shared__ float shared[TPB];

    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    float local = 0.0f;

    // grid-stride 로 모든 요소 커버 (긴 벡터 대응)
    for (int idx = globalId; idx < n; idx += blockDim.x * gridDim.x) {
        local += a[idx] * b[idx];
    }

    // 공유 메모리에 thread 별 누적값 저장
    shared[threadIdx.x] = local;
    __syncthreads();

    // 블록 내 reduction (절반씩 접는 방식)
    // TPB 가 2의 거듭제곱이라는 가정 하에서 동작
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 블록의 대표(threadIdx.x == 0)가 부분합을 기록
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

// 부분합 배열을 하나로 합치는 간단한 커널 (원자연산 사용)
__global__ void finalize_with_atomic(const float *partial_sums, float *result, int num_partials) {
    float sum = 0.0f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_partials; i += blockDim.x * gridDim.x) {
        sum += partial_sums[i];
    }
    // 블록 내 reduction 없이, 각 thread 의 sum 을 원자적으로 누적
    atomicAdd(result, sum);
}

static float dot_cpu_ref(const float *a, const float *b, int n) {
    double acc = 0.0; // 정확도 향상 위해 double 누적
    for (int i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(acc);
}

int main() {
    std::printf("[dot_product_shared] N=%d, TPB=%d\n", N, TPB);

    // 1) 호스트 데이터 준비
    float *h_a = (float*)std::malloc(sizeof(float) * N);
    float *h_b = (float*)std::malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;          // 간단한 검증을 위해 1.0
        h_b[i] = (i % 3) - 1.0f; // -1, 0, 1 반복
    }

    float ref = dot_cpu_ref(h_a, h_b, N);

    // 2) 디바이스 메모리 할당/복사
    float *d_a = nullptr, *d_b = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_a, sizeof(float) * N));
    HANDLE_ERROR(cudaMalloc((void**)&d_b, sizeof(float) * N));
    HANDLE_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));

    // 3) 부분합/결과 버퍼 준비
    int numBlocks = 512; // 충분한 블록 수로 전체 범위 커버 (필요시 -D 로 조정)
    float *d_partials = nullptr;
    float *d_result = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_partials, sizeof(float) * numBlocks));
    HANDLE_ERROR(cudaMalloc((void**)&d_result, sizeof(float)));
    HANDLE_ERROR(cudaMemset(d_result, 0, sizeof(float)));

    // 4) 1단계: 블록별 reduction 으로 부분합 계산
    dot_block_reduce_kernel<<<numBlocks, TPB>>>(d_a, d_b, d_partials, N);
    SYNC_AND_CHECK();

    // 5) 2단계: 부분합을 하나의 결과로 합치기 (원자연산)
    //   - 소규모 배열이므로 간단히 하나의 커널로 처리
    int finalizeBlocks = 1;
    int finalizeThreads = 256;
    finalize_with_atomic<<<finalizeBlocks, finalizeThreads>>>(d_partials, d_result, numBlocks);
    SYNC_AND_CHECK();

    float h_result = 0.0f;
    HANDLE_ERROR(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("GPU dot = %.6f, CPU ref = %.6f, diff = %.6f\n", h_result, ref, std::abs(h_result - ref));

    // 6) 자원 해제
    HANDLE_ERROR(cudaFree(d_a));
    HANDLE_ERROR(cudaFree(d_b));
    HANDLE_ERROR(cudaFree(d_partials));
    HANDLE_ERROR(cudaFree(d_result));
    std::free(h_a);
    std::free(h_b);

    return 0;
} 