#include <cstdio>
#include "common/book.h"  // 학습용: 이후 예제들과 동일한 헤더를 사용합니다.

// 3.1 첫 번째 프로그램 (호스트 전용)
// - CUDA 프로젝트 안에서도 일반 C/C++ 코드만으로 구성된 실행 파일을 만들 수 있습니다.
// - GPU를 사용하지 않으며, CPU에서만 실행됩니다.
int main() {
    std::printf("Hello, World! (host only)\n");
    return 0;
} 