# 4장: CUDA C에서의 병렬 프로그래밍

---

## 4.1 챕터 목표 (Chapter Objectives)

* CUDA가 병렬성을 노출하는 기본 방법을 이해한다.
* CUDA C를 이용해 **첫 번째 병렬 코드**를 작성한다.

👉 즉, 단순히 GPU에서 실행시키는 것에서 그치지 않고, **실제 병렬성을 활용하는 방법**을 배우는 것이 핵심이에요.

---

## 4.2 벡터 합 (Summing Vectors)

### CPU 버전

```c
#define N 10

void add(int *a, int *b, int *c) {
    int tid = 0;    
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;   // CPU는 직렬 실행 → 1씩 증가
    }
}
```

* CPU에서는 한 번에 하나의 요소만 더하기 때문에 반복문이 필요합니다.
* 듀얼 코어 이상 CPU라면 `tid += 2` 방식으로 병렬화도 가능하지만, **스레드 관리**가 필요하므로 코드가 복잡해집니다.

👉 여기서 의도적으로 while 루프를 사용한 이유는, 나중에 **GPU 스레드 병렬화**로 연결하기 위함이에요.

---

### GPU 버전

```c
__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;    // block index 사용
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void) {
    // CPU 메모리 준비
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // GPU 메모리 할당
    cudaMalloc((void**)&dev_a, N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*sizeof(int));

    // CPU에서 데이터 초기화 후 GPU로 복사
    for (int i=0; i<N; i++) { a[i] = -i; b[i] = i*i; }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    // GPU에서 N개의 병렬 블록 실행
    add<<<N,1>>>(dev_a, dev_b, dev_c);

    // 결과 복사
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    // 출력
    for (int i=0; i<N; i++) printf("%d + %d = %d\n", a[i], b[i], c[i]);

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 0;
}
```

### 핵심 개념

1. **`<<<N,1>>>`**

   * `N`개의 block을 실행한다는 의미.
   * `blockIdx.x` 값이 0\~N-1을 자동으로 가져와 각 요소를 담당.

2. **CUDA 내장 변수**

   * `blockIdx.x` : 현재 실행 중인 block의 index.
   * `gridDim.x` : 전체 grid 크기.
   * `threadIdx.x` : block 내부의 thread index.

👉 여기서는 block당 thread가 1개(`<<<N,1>>>`)라 `blockIdx.x`만으로 충분합니다.

---

## 4.3 병렬 개념 심화

* GPU는 **grid(격자)와 block(블록)** 구조로 스레드를 관리합니다.
* 하나의 grid는 여러 block으로 구성되고, block 내부에는 여러 thread가 존재합니다.

예:

```c
kernel<<<gridDim, blockDim>>>(...);
```

* `gridDim` : 몇 개의 block을 띄울지 (grid 크기)
* `blockDim` : 각 block 안에 몇 개의 thread를 둘지

👉 즉, **전체 스레드 수 = gridDim × blockDim**

---

## 4.4 Julia Set 예제 (Fractal)

> "벡터 합"은 너무 단순하니, CUDA 병렬성을 더 잘 보여주기 위해 \*\*Julia Set (프랙탈 그림)\*\*을 그리는 예제가 등장합니다.

### CPU 버전

```c
int julia(int x, int y) {
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    for (int i=0; i<200; i++) {
        a = a*a + c;
        if (a.magnitude2() > 1000) return 0;
    }
    return 1;
}
```

* 반복적으로 `z = z^2 + c`를 계산하여 발산 여부 판단.
* 발산하면 0, bounded 하면 1 → 픽셀 색상 결정.

### GPU 버전

```c
__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    int juliaValue = julia(x, y);

    ptr[offset*4 + 0] = 255 * juliaValue; // R
    ptr[offset*4 + 1] = 0;                // G
    ptr[offset*4 + 2] = 0;                // B
    ptr[offset*4 + 3] = 255;              // Alpha
}
```

* grid를 2차원(`dim3 grid(DIM, DIM)`)으로 실행 → 픽셀 단위 병렬 연산.
* block이 (x,y) 좌표를 직접 담당하므로 for 루프 불필요.

---

## 4.5 보충 설명 (제가 추가한 해설)

1. **GPU 병렬화의 장점**

   * CPU: 한 루프에서 N번 반복.
   * GPU: 스레드가 N개 동시에 실행 → `O(N)`에서 `O(1)`로 바뀌는 효과.

2. **안전성 체크 (`if (tid < N)`)**

   * launch할 때 항상 N개의 스레드를 쓰진 않음 → 범위 초과 방지용.
   * 잘못된 메모리 접근은 GPU에서 **커널 크래시** 유발 가능.

3. **실습 팁**

   * `#define N 10`을 `1000000`으로 바꿔보면 성능 차이가 확연히 보임.
   * Julia Set 예제에서는 `c` 값이나 `scale` 값을 바꿔보면 **다양한 프랙탈 패턴**을 얻을 수 있음.

---

✅ 정리하면,
4장은 **CPU 직렬 루프 → GPU 병렬 스레드**로의 전환을 가장 직관적으로 보여주는 챕터입니다.

* 벡터 합으로 **blockIdx / threadIdx 기본 개념**을 익히고,
* Julia Set으로 **2D grid 병렬 연산**을 시각적으로 경험하게 해주죠.
