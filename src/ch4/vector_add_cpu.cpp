#include <cstdio>

// 벡터 길이 (학습을 위해 작게)
#ifndef N
#define N 10
#endif

// CPU 버전 벡터 합
// - 의도적으로 while 루프를 사용하여 tid 를 1씩 증가시키며 직렬로 수행합니다.
// - 나중에 GPU 스레드 병렬화와의 대비를 위해 구조를 맞춥니다.
static void add(const int *a, const int *b, int *c) {
    int tid = 0; // 현재 처리할 인덱스
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1; // CPU 직렬 실행 → 1씩 증가
    }
}

int main() {
    int a[N], b[N], c[N];

    // 데이터 초기화: A = -i, B = i*i
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }

    // CPU에서 계산 수행
    add(a, b, c);

    // 결과 출력
    for (int i = 0; i < N; ++i) {
        std::printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
} 