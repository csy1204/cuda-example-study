#include <cstdio>
#include "common/book.h"

// 3.2 커널 호출 — GPU에서 빈 함수를 실행해보기
// 핵심 포인트
// - __global__ 로 표시된 함수는 GPU(디바이스)에서 실행됩니다.
// - kernel<<<gridDim, blockDim>>>(); 문법으로 호스트에서 실행을 지시합니다.
// - 커널 런치는 기본적으로 비동기입니다. 필요 시 동기화로 완료를 보장합니다.

__global__ void kernel() {
    // 학습을 위해 일부러 아무 것도 하지 않는 빈 커널
}

int main() {
    // 1개의 블록(block), 블록당 1개의 스레드(thread)로 커널을 실행합니다.
    kernel<<<1, 1>>>();

    // 디버깅 시에는 커널이 끝나길 기다리고, 에러를 즉시 확인하는 습관이 좋습니다.
    SYNC_AND_CHECK();

    std::printf("Hello, World! (after empty kernel)\n");
    return 0;
} 