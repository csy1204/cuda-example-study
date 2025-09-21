# ch4 - CUDA C에서의 병렬 프로그래밍 예제

이 디렉터리는 `docs/ch4.md` 내용을 실습하기 위한 두 가지 CUDA 예제를 포함합니다.

- `vector_add.cu`: 벡터 합 (blockIdx.x 를 이용한 1D 병렬화)
- `julia_ppm.cu`: Julia Set 프랙탈을 PPM 이미지로 렌더링 (2D grid / 2D block)

## 요구 사항

- CUDA Toolkit (nvcc 포함)
- NVIDIA GPU 및 드라이버

## 빌드

```bash
cd src/ch4
make
```

성공하면 실행 파일이 `src/ch4/bin` 에 생성됩니다.

## 실행

### 1) 벡터 합 (GPU)

```bash
make run_vector_add
```

### 1-1) 벡터 합 (CPU)

```bash
make run_vector_add_cpu
```

### 2) Julia Set 렌더링 (GPU, PPM)

```bash
make run_julia_ppm
```

### 2-1) Julia Set 렌더링 (CPU, PPM)

```bash
make run_julia_cpu_ppm
```

현재 디렉터리에 `julia.ppm` 또는 `julia_cpu.ppm` 파일이 생성됩니다. PPM 을 지원하는 뷰어로 열어 확인하세요.

- 이미지 크기, 반복 횟수, 상수 c 는 소스 코드 상단의 매크로로 쉽게 조정할 수 있습니다.
  - `IMG_WIDTH`, `IMG_HEIGHT`, `MAX_ITER`, `JULIA_C_RE`, `JULIA_C_IM`
  - 예: `make NVCCFLAGS="-O2 -DIMG_WIDTH=800 -DIMG_HEIGHT=600 -DMAX_ITER=300"`

## 학습 포인트

- `vector_add.cu` 는 `<<<N,1>>>` 구성에서 `blockIdx.x` 로 각 요소를 담당하는 가장 단순한 병렬화 방식을 보여줍니다.
- `julia_ppm.cu` 는 2D 블록과 2D 그리드를 사용하여 픽셀 단위로 병렬 연산을 수행합니다.
- 두 예제 모두 Host(Device 메모리 할당/복사) ↔ Device(커널 실행) 간 데이터 흐름을 명확히 보여주며, 안전한 커널 실행을 위해 `cudaDeviceSynchronize()` 와 에러 체크를 포함합니다. 