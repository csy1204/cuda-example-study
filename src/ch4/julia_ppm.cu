#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// 간단한 에러 체크 매크로 (학습용)
#define cudaCheck(stmt)                                                                          \
    do {                                                                                         \
        cudaError_t err__ = (stmt);                                                              \
        if (err__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA Error %s at %s:%d -> %s\n", #stmt, __FILE__, __LINE__, \
                        cudaGetErrorString(err__));                                              \
            std::exit(EXIT_FAILURE);                                                             \
        }                                                                                        \
    } while (0)

// 출력 이미지 크기
#ifndef IMG_WIDTH
#define IMG_WIDTH  1024*2
#endif
#ifndef IMG_HEIGHT
#define IMG_HEIGHT 1024*2
#endif

// 최대 반복 횟수 (Julia 수렴/발산 판별)
#ifndef MAX_ITER
#define MAX_ITER 1000
#endif

// Julia Set 상수 c = cRe + i*cIm
#ifndef JULIA_C_RE
#define JULIA_C_RE -0.8f
#endif
#ifndef JULIA_C_IM
#define JULIA_C_IM  0.156f
#endif

// 간단한 복소수 타입 (Device 전용 연산 포함)
struct Complex {
    float real;
    float imag;

    __device__ Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}

    __device__ float magnitude2() const { return real * real + imag * imag; }

    __device__ Complex operator*(const Complex &other) const {
        return Complex(real * other.real - imag * other.imag,
                        imag * other.real + real * other.imag);
    }

    __device__ Complex operator+(const Complex &other) const {
        return Complex(real + other.real, imag + other.imag);
    }
};

// 한 픽셀에 대한 Julia 판정: 최대 반복 내에 발산하지 않으면 1, 발산하면 0 반환
__device__ int juliaAt(int x, int y, int width, int height, float cRe, float cIm, int maxIter) {
    // 화면 좌표 (x,y)를 복소 평면 좌표 (jx, jy) 로 매핑
    // - 화면 중심을 (0,0) 으로, 대략 [-1.5, 1.5] 범위로 정규화
    float jx = 1.5f * ((x - width / 2.0f) / (width / 2.0f));
    float jy = 1.5f * ((y - height / 2.0f) / (height / 2.0f));

    Complex a(jx, jy);
    Complex c(cRe, cIm);

    for (int i = 0; i < maxIter; ++i) {
        a = a * a + c;
        if (a.magnitude2() > 1000.0f) { // 충분히 멀어지면 발산으로 판단
            return 0;                    // 발산 → 바깥 영역
        }
    }
    return 1; // 반복 내 발산하지 않음 → 집합 내부
}

// 픽셀 단위 병렬 커널: PPM(RGB) 버퍼를 채움
__global__ void renderJulia(unsigned char *rgb, int width, int height,
                            float cRe, float cIm, int maxIter) {
    // 전형적인 2D 인덱싱 (blockDim/Idx + threadDim/Idx)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int inside = juliaAt(x, y, width, height, cRe, cIm, maxIter);

    // PPM 은 3바이트 RGB 포맷
    int offset = (y * width + x) * 3;

    // 단순 색상 매핑 (책의 예: 내부는 빨강, 외부는 검정)
    rgb[offset + 0] = static_cast<unsigned char>(255 * inside); // R
    rgb[offset + 1] = 0;                                        // G
    rgb[offset + 2] = 0;                                        // B
}

// 간단한 PPM 저장 (binary P6)
static void savePPM(const char *filepath, const unsigned char *rgb, int width, int height) {
    std::FILE *fp = std::fopen(filepath, "wb");
    if (!fp) {
        std::perror("fopen");
        std::fprintf(stderr, "Failed to open output file: %s\n", filepath);
        std::exit(EXIT_FAILURE);
    }
    // 헤더: P6\n<width> <height>\n255\n
    std::fprintf(fp, "P6\n%d %d\n255\n", width, height);
    size_t totalBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    size_t wrote = std::fwrite(rgb, 1, totalBytes, fp);
    if (wrote != totalBytes) {
        std::fprintf(stderr, "PPM write failed (wrote %zu / %zu bytes)\n", wrote, totalBytes);
        std::fclose(fp);
        std::exit(EXIT_FAILURE);
    }
    std::fclose(fp);
}

int main() {
    const int width = IMG_WIDTH;
    const int height = IMG_HEIGHT;
    const char *outPath = "julia.ppm"; // 출력 파일명

    // 1) GPU 버퍼 할당 (width * height * 3 바이트)
    unsigned char *devRGB = nullptr;
    size_t numBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&devRGB), numBytes));

    // 2) 그리드/블록 구성
    //    - 2D 블록(16x16), 2D 그리드로 전형적인 CUDA 병렬화를 구성
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 3) 커널 실행
    renderJulia<<<gridDim, blockDim>>>(devRGB, width, height, JULIA_C_RE, JULIA_C_IM, MAX_ITER);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // 4) Host 로 결과 복사 후 파일 저장
    unsigned char *hostRGB = static_cast<unsigned char *>(std::malloc(numBytes));
    if (!hostRGB) {
        std::fprintf(stderr, "malloc failed for hostRGB (%zu bytes)\n", numBytes);
        std::exit(EXIT_FAILURE);
    }
    cudaCheck(cudaMemcpy(hostRGB, devRGB, numBytes, cudaMemcpyDeviceToHost));

    savePPM(outPath, hostRGB, width, height);
    std::printf("[OK] PPM saved: %s (%dx%d)\n", outPath, width, height);

    // 5) 정리
    std::free(hostRGB);
    cudaCheck(cudaFree(devRGB));

    return 0;
} 