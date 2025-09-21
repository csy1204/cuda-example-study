#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "../ch3/common/book.h"

#ifndef DIM
#define DIM 512  // 이미지 한 변 길이 (DIM x DIM)
#endif

// 픽셀 1개를 담당하는 스레드가 그레이스케일 값을 계산하여 RGBA 버퍼에 기록합니다.
__global__ void ripple_kernel(unsigned char *rgba, int ticks) {
    int x = threadIdx.x + blockIdx.x * blockDim.x; // 전역 x 좌표
    int y = threadIdx.y + blockIdx.y * blockDim.y; // 전역 y 좌표

    if (x >= DIM || y >= DIM) return; // 범위 체크

    int offset = x + y * (blockDim.x * gridDim.x); // 선형 인덱스

    float fx = x - DIM / 2.0f;
    float fy = y - DIM / 2.0f;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                          cosf(d / 10.0f - ticks / 7.0f) /
                          (d / 10.0f + 1.0f));

    rgba[offset * 4 + 0] = grey;
    rgba[offset * 4 + 1] = grey;
    rgba[offset * 4 + 2] = grey;
    rgba[offset * 4 + 3] = 255; // alpha
}

// 간단한 PPM(Binary P6) 파일로 저장
static void save_ppm_rgba_as_grey_p6(const char *filename, const unsigned char *rgba, int width, int height) {
    FILE *fp = std::fopen(filename, "wb");
    if (!fp) {
        std::perror("fopen");
        std::exit(EXIT_FAILURE);
    }
    std::fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // RGBA 를 RGB 로 변환하여 저장 (alpha 무시)
    for (int i = 0; i < width * height; ++i) {
        unsigned char rgb[3];
        rgb[0] = rgba[i * 4 + 0];
        rgb[1] = rgba[i * 4 + 1];
        rgb[2] = rgba[i * 4 + 2];
        std::fwrite(rgb, 1, 3, fp);
    }
    std::fclose(fp);
}

int main() {
    std::printf("[ripple_ppm] DIM=%d\n", DIM);

    const int width = DIM;
    const int height = DIM;
    const int numPixels = width * height;
    const size_t rgbaBytes = static_cast<size_t>(numPixels) * 4u;

    // 1) 디바이스 버퍼 할당 (RGBA)
    unsigned char *d_rgba = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&d_rgba, rgbaBytes));

    // 2) 그리드/블록 구성: 16x16 스레드 블록, 2D 그리드로 전체 픽셀 커버
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    int ticks = 0; // 시간 변수 (애니메이션 프레임에서 변경 가능)

    ripple_kernel<<<blocks, threads>>>(d_rgba, ticks);
    SYNC_AND_CHECK();

    // 3) 결과를 호스트로 복사
    unsigned char *h_rgba = (unsigned char*)std::malloc(rgbaBytes);
    HANDLE_ERROR(cudaMemcpy(h_rgba, d_rgba, rgbaBytes, cudaMemcpyDeviceToHost));

    // 4) 파일 저장 (PPM)
    const char *outFile = "ripple.ppm";
    save_ppm_rgba_as_grey_p6(outFile, h_rgba, width, height);
    std::printf("Saved %s\n", outFile);

    // 5) 자원 해제
    std::free(h_rgba);
    HANDLE_ERROR(cudaFree(d_rgba));

    return 0;
} 