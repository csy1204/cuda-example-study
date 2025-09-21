# 5장: 스레드 협력 (Thread Cooperation)

---

## 5.1 챕터 목표 (Chapter Objectives)

이 장에서 배우는 것:

* CUDA C에서 “thread”의 개념 이해
* 서로 다른 thread들이 **통신하는 방법**
* 병렬 실행되는 thread들을 **동기화하는 방법**

👉 앞 장에서는 block 단위 병렬성만 다뤘지만, 여기서는 block 내부의 thread들이 어떻게 **협력(cooperate)**하는지를 배우게 됩니다.

---

## 5.2 블록을 스레드로 나누기 (Splitting Parallel Blocks)

* 이전 장에서는 `add<<<N,1>>>` 형태로 block을 여러 개 실행 → 각 block이 벡터의 한 원소를 처리.
* 하지만 CUDA는 block을 다시 여러 개의 thread로 쪼갤 수 있음.
* 즉, `add<<<1,N>>>` 실행 시 → **block 1개, thread N개**
  이 경우 각 thread가 하나의 원소를 담당.

### 코드 예시 (Vector Add - Thread 기반)

```c
__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x;   // block이 아니라 thread 인덱스 사용
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void) {
    ...
    add<<<1, N>>>(dev_a, dev_b, dev_c); // block=1, thread=N
    ...
}
```

👉 핵심: **`blockIdx.x` → `threadIdx.x`**로 바뀜.

---

## 5.3 더 긴 벡터 처리 (GPU Sums of a Longer Vector)

문제: block당 최대 thread 수(`maxThreadsPerBlock`) 제한 존재 (예: 512 또는 1024).
→ 긴 벡터를 처리하려면 **여러 block + 여러 thread** 조합 필요.

### 새로운 인덱싱 방식

```c
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

* `threadIdx.x` : block 내 위치
* `blockIdx.x * blockDim.x` : block 단위 offset
* 둘을 합쳐서 **전체 thread의 글로벌 id**를 구함.

### 실행 예시

```c
add<<<(N+127)/128, 128>>>(dev_a, dev_b, dev_c);
```

* block당 128 thread
* 총 thread 수 ≥ N이 되도록 `(N+127)/128` block 실행
* overshoot 된 thread는 `if (tid < N)` 조건으로 필터링

---

## 5.4 임의 길이 벡터 처리 (GPU Sums of Arbitrarily Long Vectors)

block 수에도 제한 있음 (`gridDim.x ≤ 65535`).
→ 벡터 길이가 매우 크면 단순 block 분할만으로는 부족.

### 해결: while 루프를 통한 stride 접근

```c
__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;  // stride jump
    }
}
```

👉 thread들이 전체 벡터를 “stride” 단위로 나눠 맡음.

이 방식은 CPU에서 다중 코어 루프를 설계할 때 “i += num_processors” 하던 것과 비슷합니다.

---

## 5.5 실습 예제: GPU Ripple 애니메이션

책에서는 단순 벡터 연산 대신 **리플(Ripple) 효과 애니메이션**을 GPU로 생성하는 예제를 제시.

### 주요 개념

* `dim3 blocks(DIM/16, DIM/16);`
* `dim3 threads(16,16);`
  → 2차원 블록, 2차원 스레드 → 한 픽셀당 1 thread 매핑.

### 커널 코드

```c
__global__ void kernel(unsigned char *ptr, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf(fx*fx + fy*fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                       cos(d/10.0f - ticks/7.0f) /
                       (d/10.0f + 1.0f));

    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}
```

👉 각 thread가 픽셀 좌표 `(x,y)`를 계산해 해당 위치 색상을 결정 → 애니메이션 프레임 생성.

---

## 5.6 공유 메모리와 동기화 (Shared Memory and Synchronization)

* **공유 메모리(shared memory)**: block 내 모든 thread가 공유하는 빠른 on-chip 메모리.
* `__shared__` 키워드로 선언.
* block 간에는 공유 불가 (block 단위 isolation).
* 이 메모리를 이용하면 thread들이 **데이터 교환, 협력 계산** 가능.

문제: thread들이 동시에 접근하면 **race condition** 발생 → 결과가 불확실.
→ 해결: **동기화 함수** `__syncthreads()` 사용.

---

## 5.7 예제: 내적 (Dot Product)

* 내적(dot product) = 두 벡터의 원소별 곱 → 합산 → 스칼라 결과.
* 단계:

  1. 각 thread가 부분 곱을 계산 후 공유 메모리에 저장.
  2. `__syncthreads()`로 모든 thread가 완료되었는지 동기화.
  3. 일부 thread가 공유 메모리 값들을 합산 (reduction) 수행.

👉 이때 **공유 메모리 + 동기화**가 thread 협력의 핵심.

---

# 💡 보충 설명 (제가 추가한 해설)

1. **스레드 협력의 의의**

   * 단순 벡터 합은 thread별 독립 계산 가능.
   * 하지만 **내적, 행렬 곱, 합산(reduction), convolution** 등은 thread 간 협력이 필수.

2. **GPU 아키텍처 고려**

   * 공유 메모리는 매우 빠르지만 크기 제한(보통 48KB/block).
   * 동기화를 남용하면 성능 저하 → 최소화 필요.

3. **실습 아이디어**

   * 벡터 합 코드를 내적으로 확장 → 공유 메모리 사용해보기.
   * 리플 애니메이션의 수식을 바꿔서 (예: `sin`, `tan`) 다른 패턴 생성해보기.

---

✅ 정리하면,
5장은 **스레드 협력의 시작**을 다루며,

* 벡터 합 → 긴 벡터 → 임의 길이 벡터
* 애니메이션 예제 (리플)
* 공유 메모리와 동기화, 내적(dot product)

까지 **병렬 thread들의 협력과 동기화 기초**를 단계별로 소개합니다.


<!-- 실습 코드 안내: src/ch5 에 실습 예제가 포함되어 있으며, `make` 및 `make run_*` 타깃으로 바로 실행할 수 있습니다. -->

