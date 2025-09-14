#include <cstdio>
#include "common/book.h"

// 3.3 매개변수 전달 & 디바이스 메모리 사용 예제
// 흐름 요약
// 1) cudaMalloc 으로 디바이스 메모리 할당
// 2) 커널 인자로 값과 디바이스 포인터 전달
// 3) cudaMemcpy 로 결과를 호스트로 복사 (동기점)
// 4) cudaFree 로 디바이스 메모리 해제

__global__ void add(int a, int b, int *out_c) {
    // 주의: 이 포인터는 디바이스 메모리 주소입니다. 디바이스 코드에서만 역참조 가능합니다.
    *out_c = a + b;
}

int main() {
    int host_c = 0;          // 호스트 측 결과 저장 변수
    int *device_c = nullptr; // 디바이스 메모리 포인터

    // 정수 1개를 저장할 디바이스 메모리 할당
    HANDLE_ERROR(cudaMalloc((void **)&device_c, sizeof(int)));

    // HANDLE_ERROR(cudaMalloc(reinterpret_cast<void **>(&device_c), sizeof(int)));

    // 값 2, 7과 디바이스 포인터 device_c 를 커널 인자로 전달
    add<<<1, 1>>>(3, 7, device_c);

    // host_c 로 결과를 복사 (Device->Host). 이 복사는 동기이며, 커널 완료를 보장합니다.
    HANDLE_ERROR(cudaMemcpy(&host_c, device_c, sizeof(int), cudaMemcpyDeviceToHost));

    // 커널 런치 시점의 런타임 에러가 있었다면 여기서 확인됩니다.
    CHECK_LAST_CUDA_ERROR();

    std::printf("3 + 7 = %d\n", host_c);

    // 자원 정리
    HANDLE_ERROR(cudaFree(device_c));

    return 0;
} 