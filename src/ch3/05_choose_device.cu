#include <cstdio>
#include <cstring>
#include "common/book.h"

// 3.5 디바이스 속성 활용 — 조건에 맞는 GPU 선택
// 개념
// - 요구 조건(예: 최소 컴퓨트 캡퍼빌리티)을 가진 디바이스를 cudaChooseDevice 로 선택
// - 이후 cudaSetDevice 로 활성화

int main() {
    int current_dev = -1;
    HANDLE_ERROR(cudaGetDevice(&current_dev));
    std::printf("ID of current CUDA device: %d\n", current_dev);

    // 최소 요구 사양 예시: compute capability >= 1.3 (책 예시)
    cudaDeviceProp desired;
    std::memset(&desired, 0, sizeof(desired));
    desired.major = 1;
    desired.minor = 3;

    int chosen_dev = -1;
    HANDLE_ERROR(cudaChooseDevice(&chosen_dev, &desired));
    std::printf("ID of CUDA device closest to revision %d.%d: %d\n",
                desired.major, desired.minor, chosen_dev);

    // 선택한 디바이스를 현재 컨텍스트로 설정
    HANDLE_ERROR(cudaSetDevice(chosen_dev));

    // 확인 출력
    int verify = -1;
    HANDLE_ERROR(cudaGetDevice(&verify));
    std::printf("Active CUDA device is now: %d\n", verify);

    return 0;
} 