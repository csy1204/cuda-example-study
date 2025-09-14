#include <stdio.h>
#include "common/book.h" // 에러 핸들링 매크로를 위함

// TODO 1: 입력받은 정수 'a'를 제곱하여 포인터 'result'가 가리키는 곳에
// 저장하는 커널 함수를 '__global__' 키워드를 사용하여 정의하세요.
// 함수 이름은 'square_kernel'로 합니다.

__global__ void square_kernel(int a, int *result) {
    *result = a * a;
}

int main(void) {
    int h_a = 9;      // CPU(호스트)에 있는 입력값
    int h_result;   // CPU(호스트)에 저장될 결과값
    int *d_result;  // GPU(디바이스) 메모리를 가리킬 포인터

    // TODO 2: 정수 하나를 저장할 공간을 GPU 메모리에 할당하고,
    // 그 주소를 d_result 포인터에 저장하세요.
    // cudaMalloc()을 사용합니다.
    HANDLE_ERROR(cudaMalloc((void **)&d_result, sizeof(int)));

    // TODO 3: 위에서 정의한 square_kernel 커널을 <<<1, 1>>> 구성으로
    // 호출하세요. h_a와 d_result를 인자로 전달합니다.
    square_kernel<<<1, 1>>>(h_a, d_result);

    // TODO 4: GPU 메모리(d_result)에 저장된 결과값을
    // CPU 변수(h_result)로 복사해오세요.
    // cudaMemcpy()와 cudaMemcpyDeviceToHost를 사용합니다.
    HANDLE_ERROR(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("%d의 제곱은 %d 입니다.\n", h_a, h_result);
    // TODO 5: 할당했던 GPU 메모리를 해제하세요.
    // cudaFree()를 사용합니다.
    HANDLE_ERROR(cudaFree(d_result));

    CHECK_LAST_CUDA_ERROR();

    return 0;
}