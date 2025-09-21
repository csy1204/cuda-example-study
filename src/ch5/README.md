# ch5 - 스레드 협력(Thread Cooperation) 실습 예제

이 디렉터리는 `docs/ch5.md` 내용을 학습용 코드로 구현한 예제를 포함합니다.

## 포함된 예제
- vector_add_threads.cu: `add<<<1, N>>>` — 한 블록 안에 N개의 스레드를 사용하여 벡터 합을 수행합니다. (threadIdx.x 사용)
- vector_add_1d_grid.cu: `add<<<(N+TPB-1)/TPB, TPB>>>` — 1D 그리드/블록 구성으로 긴 벡터를 처리합니다. (글로벌 인덱스)
- vector_add_stride.cu: while-루프 기반 stride 접근 — 매우 긴 벡터를 처리합니다. (gridDim.x, blockDim.x 활용)
- ripple_ppm.cu: 2D 그리드/블록을 이용해 리플(Ripple) 효과 이미지를 생성하여 PPM으로 저장합니다.
- dot_product_shared.cu: 공유 메모리와 `__syncthreads()`를 이용한 내적(reduction) 예제.

## 요구 사항
- CUDA Toolkit (nvcc 포함)
- NVIDIA GPU 및 드라이버

## 빌드
```bash
cd src/ch5
make
```

성공하면 실행 파일이 `src/ch5/bin` 에 생성됩니다.

## 실행
- 벡터 합 (한 블록, N 스레드)
```bash
make run_vector_add_threads
```
- 벡터 합 (1D 그리드/블록)
```bash
make run_vector_add_1d_grid
```
- 벡터 합 (stride 방식)
```bash
make run_vector_add_stride
```
- 리플 애니메이션 프레임 PPM 생성
```bash
make run_ripple_ppm
```
- 공유 메모리 내적
```bash
make run_dot_product_shared
```

## 참고
- 모든 예제는 `src/ch3/common/book.h`의 에러 처리 도우미를 사용합니다.
- 매크로 파라미터(N, DIM, TPB 등)는 빌드 시 오버라이드할 수 있습니다. 예:
```bash
make NVCCFLAGS="-O2 -DN=2048 -DTPB=256"
``` 