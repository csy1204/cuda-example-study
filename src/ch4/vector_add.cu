#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 간단한 에러 체크 매크로 (학습용)
#define cudaCheck(stmt)                                                                          \
    do {                                                                                         \
        cudaError_t err__ = (stmt);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA Error %s at %s:%d -> %s\n", #stmt, __FILE__, __LINE__, \
                        cudaGetErrorString(err__));                                              \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

// 벡터 길이 (학습을 위해 작게)
#define N 10

// GPU에서 실행되는 커널 함수
// - 여기서는 block 당 thread 수를 1개로 두고(blockDim=1), blockIdx.x 만을 이용해 인덱싱합니다.
// - 즉, 각 block 이 벡터의 한 요소를 담당합니다.
__global__ void add(const int *a, const int *b, int *c) {
    int tid = blockIdx.x; // 현재 block 의 x 인덱스 (0 ~ gridDim.x-1)
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // 1) CPU 메모리 준비 (Host)
    int hostA[N], hostB[N], hostC[N];

    // 데이터 초기화: A = -i, B = i*i
    for (int i = 0; i < N; ++i) {
        hostA[i] = -i;
        hostB[i] = i * i;
    }

    // 2) GPU 메모리 할당 (Device)
    int *devA = nullptr;
    int *devB = nullptr;
    int *devC = nullptr;

    cudaCheck(cudaMalloc((void**)&devA, N * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&devB, N * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&devC, N * sizeof(int)));

    // 3) Host -> Device 복사
    cudaCheck(cudaMemcpy(devA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devB, hostB, N * sizeof(int), cudaMemcpyHostToDevice));

    // 4) 커널 실행
    //    각 block 이 벡터의 한 원소를 담당하도록 gridDim=N, blockDim=1 로 실행합니다.
    add<<<N, 1>>>(devA, devB, devC);

    // 커널 런치 에러 확인 및 동기화
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // 5) Device -> Host 복사
    cudaCheck(cudaMemcpy(hostC, devC, N * sizeof(int), cudaMemcpyDeviceToHost));

    // 6) 결과 출력
    for (int i = 0; i < N; ++i) {
        std::printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i]);
    }

    // 7) 자원 해제
    cudaCheck(cudaFree(devA));
    cudaCheck(cudaFree(devB));
    cudaCheck(cudaFree(devC));

    return 0;
} 