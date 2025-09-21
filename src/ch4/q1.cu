#include <cuda.h>
#include <stdio.h>

#define N 1000

// TODO: GPU에서 실행할 커널 함수를 작성하세요
__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x; // 현재 스레드의 인덱스를 구하세요

  if (tid < N) {
    // TODO: 벡터 덧셈 연산을 구현하세요
    c[tid] = a[tid] + b[tid];
  }
}

int main(void) {
  // CPU 메모리
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  // TODO: GPU 메모리 할당 (3개의 배열)
  cudaMalloc(&dev_a, N * sizeof(int));
  cudaMalloc(&dev_b, N * sizeof(int));
  cudaMalloc(&dev_c, N * sizeof(int));

  // 데이터 초기화
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // TODO: CPU에서 GPU로 데이터 복사 (a, b 배열)
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // TODO: 커널 실행 (N개의 블록, 각 블록당 1개의 스레드)
  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  // TODO: GPU에서 CPU로 결과 복사
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  // 결과 확인 (처음 10개만)
  printf("결과 확인:\n");
  for (int i = 0; i < 10; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  // TODO: GPU 메모리 해제
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}