## 3장 목표 (Chapter Objectives)

* **첫 CUDA C 코드 작성**
  CUDA C의 기본 문법과 샘플 코드를 통해 “호스트(host) vs 디바이스(device)” 실행 환경 차이를 이해합니다.&#x20;
* **호스트 코드에서 디바이스 코드(커널) 실행**
  `__global__` 커널과 `<<<...>>>` 런치 구문을 사용해 GPU에서 함수를 실행하는 법을 익힙니다.&#x20;
* **디바이스 메모리 사용법**
  `cudaMalloc / cudaMemcpy / cudaFree`를 이용한 메모리 할당·복사·해제 흐름을 배웁니다.&#x20;
* **시스템의 CUDA 디바이스 정보 조회**
  `cudaGetDeviceCount`, `cudaGetDeviceProperties`로 GPU 수와 스펙을 확인합니다.&#x20;

## 3.1 첫 번째 프로그램 (A First Program) — 개념 맛보기

> 핵심: **기존 C와 똑같이 보이는 코드도 CUDA 툴체인으로 빌드**할 수 있습니다. (호스트 전용 코드)&#x20;

### 번역 & 설명

가장 단순한 “Hello, World!” 예제는 **오직 호스트(CPU)** 에서만 실행됩니다. 이 예제는 “CUDA C라고 해서 늘 특별한 건 아니다”라는 메시지를 줍니다. 즉, CUDA 프로젝트 안에서도 **일반 C 코드는 그대로** 돌아갑니다.&#x20;

### 예시 코드 (호스트 전용)

```c
#include "../common/book.h"

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
```

### 보충설명 (실무 팁)

* CUDA 프로젝트라도 **호스트 전용 유틸/로깅/파서** 등은 평범한 C/C++로 작성·컴파일됩니다.
* 빌드 도구 `nvcc`는 **호스트 코드와 디바이스 코드를 분리**해 각각 알맞은 컴파일러로 넘깁니다.&#x20;

## 3.2 커널 호출 (A Kernel Call) — GPU에서 함수 실행

> 핵심: `__global__`로 **디바이스에서 실행될 함수(커널)** 를 정의하고, `kernel<<<grid, block>>>(args...)`로 **호스트에서 실행을 지시**합니다.&#x20;

### 번역 & 설명

* `__global__` 한정자는 “이 함수는 **디바이스에서 실행**된다”는 의미입니다.
* `kernel<<<1,1>>>();`는 **런치 구성(그리드/블록 크기)** 을 런타임에 전달하는 **특별한 호출 문법**입니다.
* 커널의 **인자(argument)** 는 일반 함수처럼 괄호 `()`로 전달합니다(런치 구성은 `<<< >>>`에).&#x20;

### 예시 코드

```c
#include <iostream>

__global__ void kernel(void) { 
    // 아직 할 일 없음
}

int main(void) {
    kernel<<<1,1>>>();              // GPU에 빈 커널 실행 지시
    printf("Hello, World!\n");      // 호스트 출력
    return 0;
}
```

### 보충설명 (실무 팁/주의)

* **런치 구성**: `<<<gridDim, blockDim>>>`에서 `gridDim`은 블록 개수, `blockDim`은 블록당 스레드 수입니다. (3D 구성도 가능)
* **비동기 실행**: 커널 런치는 기본적으로 **비동기**입니다. 커널 완료를 기다리려면 `cudaDeviceSynchronize()`를 호출하거나, **호스트로의 동기 복사**(`cudaMemcpyDeviceToHost`)가 암묵적 동기점이 되기도 합니다.
* **디버깅**: 커널 내부 `printf`는 가능하지만, 성능·버퍼링을 고려하세요.

## 3.3 매개변수 전달 & 디바이스 메모리 (Passing Parameters)

> 핵심: **커널 인자 전달은 일반 함수와 동일**, 단 **디바이스 메모리는 별도 API**로 관리합니다.&#x20;

### 번역 & 설명

* 커널에 값/포인터를 인자로 **그대로** 전달할 수 있습니다.
* 유의점: **호스트에서 `cudaMalloc`으로 받은 포인터는 디바이스 주소**입니다. 이를 호스트에서 **역참조하면 안 됩니다.**
* 메모리 복사는 `cudaMemcpy(dst, src, size, kind)`에서 `kind`로 **방향**(H2D/D2H/D2D)을 명시합니다.&#x20;

### 예시 코드 (덧셈 커널)

```c
#include <iostream>
#include "book.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(void) {
    int c;
    int *dev_c;

    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

    add<<<1,1>>>(2, 7, dev_c);

    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int),
                            cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c);
    return 0;
}
```

### 포인터 사용 규칙 (원문 요지)

* `cudaMalloc` 포인터는 **디바이스 코드에서 사용 가능**(읽기/쓰기 OK).
* 같은 포인터를 **호스트 코드에서 역참조하면 안 됨**(주소공간 다름).
* 호스트 포인터는 **디바이스 코드에서 역참조 불가**. (필요시 복사로 전달)&#x20;

### 보충설명 (실무 팁/주의)

* **동기/비동기 복사**: 기본 `cudaMemcpy`는 **동기**입니다(완료까지 대기). 스트림·비동기 API(`cudaMemcpyAsync`)를 쓰면 겹치기(Overlap)가 가능합니다.
* **Pinned(페이지록) 메모리**를 쓰면 복사 대역폭을 높일 수 있습니다.
* `HANDLE_ERROR` 같은 매크로는 **학습용**엔 편리하지만, 실전엔 **명시적 에러 처리/리트라이/로깅**이 권장됩니다.&#x20;

## 3.4 디바이스 조회 (Querying Devices) — GPU 스펙 읽기

> 핵심: `cudaGetDeviceCount`로 **디바이스 수**, `cudaGetDeviceProperties`로 **각 디바이스 속성**을 확인합니다.&#x20;

### 번역 & 설명

여러 GPU가 있는 시스템에서 “**어떤 GPU가 무엇인지**”를 알아야 합니다. `cudaDeviceProp` 구조체에는 이름, 글로벌 메모리 크기, 공유 메모리, 레지스터 수, warp 크기, 최대 스레드/그리드 차원, 컴퓨트 캐퍼빌리티(major/minor), SM 수, 동시 커널 실행 가능 여부 등 다양한 정보가 들어 있습니다.&#x20;

### 예시 코드 (개수 조회)

```c
int count;
HANDLE_ERROR(cudaGetDeviceCount(&count));
```

### 예시 코드 (속성 조회·출력)

```c
#include "../common/book.h"

int main(void) {
    cudaDeviceProp prop;
    int count;

    HANDLE_ERROR(cudaGetDeviceCount(&count));

    for (int i = 0; i < count; i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

        printf("--- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: %s\n",
               prop.deviceOverlap ? "Enabled" : "Disabled");
        printf("Kernel execution timeout: %s\n",
               prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        printf("--- Memory Information ---\n");
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);

        printf("--- MP Information ---\n");
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per block: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}
```

### 보충설명 (실무 팁)

* **compute capability(예: 8.0, 9.0 등)** 은 **아키텍처 세대/기능 세트**를 의미합니다. 예를 들어 **더 높은 버전일수록** 새로운 명령/성능 최적화 기능을 제공하는 경향이 있습니다.
* **SM 수(multiProcessorCount)**, **워프 크기(warpSize)**, **최대 스레드 수** 등은 **최대 병렬도와 최적 배치**에 직접적인 영향을 줍니다.&#x20;

## 3.5 디바이스 속성 활용 (Using Device Properties) — 조건에 맞는 GPU 선택

> 핵심: `cudaChooseDevice`로 **원하는 조건(예: 최소 compute capability)** 에 맞는 디바이스를 자동 선택한 뒤 `cudaSetDevice`로 활성화합니다.&#x20;

### 번역 & 설명

예를 들어 **double 정밀도**가 필요한 경우, **compute capability ≥ 1.3**(책 기준)을 요구하도록 `cudaDeviceProp` 일부 필드를 세팅하고 `cudaChooseDevice`를 호출해 조건을 만족하는 디바이스를 찾을 수 있습니다. 이후 `cudaSetDevice(dev)`로 해당 GPU를 현재 컨텍스트로 설정합니다.&#x20;

### 예시 코드

```c
#include "../common/book.h"

int main(void) {
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;     // 최소 요구 사양(책 예시)
    prop.minor = 3;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);

    HANDLE_ERROR(cudaSetDevice(dev));
}
```

### 보충설명 (실무 팁/주의)

* **여러 GPU 시스템**에서는 “가장 빠른 GPU” 또는 “특정 기능 지원 GPU”를 **명시적으로 선택**하는 습관이 중요합니다. 런타임이 항상 최적을 택하는 보장은 없습니다.&#x20;
* 최신 CUDA에선 지원 필드/값이 확장되었습니다(예: L2 크기, 메모리 클럭, 동시 엔진 수 등). **정확한 요구 조건**을 정해두고 선택 로직을 작성하세요.

## 3.6 정리 (Chapter Review)

* CUDA C는 **기존 C에 디바이스 실행을 위한 한정자/런치 문법을 더한 형태**입니다. `__global__`과 `<<<...>>>`로 **GPU 커널**을 실행합니다.&#x20;
* 디바이스 메모리는 `cudaMalloc / cudaMemcpy / cudaFree`로 **명시적으로 관리**합니다. 포인터의 **주소공간(호스트/디바이스) 혼동 금지**가 핵심입니다.&#x20;
* `cudaGetDeviceCount / cudaGetDeviceProperties / cudaChooseDevice / cudaSetDevice`로 **GPU 스펙을 조회/선택**할 수 있습니다. 실무에선 **요구조건(컴퓨트 캡퍼빌리티·SM 수·메모리 용량 등)** 기반의 선택 로직이 중요합니다.&#x20;

## 추가로 같이 보면 좋은 보충설명

* **런치 구성 튜닝의 출발점**
  `<<<grid, block>>>`의 `block`은 보통 **32의 배수(워프 정렬)** 로 잡고, **SM 당 충분한 활성화 블록 수**를 확보해 **자원 점유율(occupancy)** 를 올리는 게 기본 전략입니다.
* **동기점과 디버깅**
  커널 실패 원인 파악이 필요할 땐 `cudaDeviceSynchronize()` 후 `cudaGetLastError()`를 확인하는 패턴이 유용합니다.
* **데이터 이동 최소화**
  가능하면 **호스트↔디바이스 복사 횟수/용량을 줄이는 설계**(배치 처리, 중간 결과 디바이스 유지)가 성능에 결정적입니다.
* **현대 CUDA 관점**
  책의 예시는 고전 API 감각으로 훌륭한 입문 가이드입니다. 실제 프로젝트에선 **스트림/이벤트**, **비동기 복사**, **유니파이드 메모리**(필요 시) 등을 추가로 사용해 **오버랩**과 **개발 편의**를 얻습니다.
