#include <cstdio>
#include <cstdlib>

// 출력 이미지 크기 (원하면 값만 바꿔 보세요)
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

// Julia Set 상수 c = cRe + i*cIm (책 예제와 동일한 값)
#ifndef JULIA_C_RE
#define JULIA_C_RE -0.8f
#endif
#ifndef JULIA_C_IM
#define JULIA_C_IM  0.156f
#endif

struct Complex {
    float real;
    float imag;
    Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}
    float magnitude2() const { return real * real + imag * imag; }
    Complex operator*(const Complex &o) const {
        return Complex(real * o.real - imag * o.imag,
                       imag * o.real + real * o.imag);
    }
    Complex operator+(const Complex &o) const { return Complex(real + o.real, imag + o.imag); }
};

// 한 픽셀에 대한 Julia 판정: 최대 반복 내에 발산하지 않으면 1, 발산하면 0 반환
static int juliaAt(int x, int y, int width, int height, float cRe, float cIm, int maxIter) {
    // 화면 좌표 (x,y)를 복소 평면 좌표 (jx, jy) 로 매핑
    float jx = 1.5f * ((x - width / 2.0f) / (width / 2.0f));
    float jy = 1.5f * ((y - height / 2.0f) / (height / 2.0f));

    Complex a(jx, jy);
    Complex c(cRe, cIm);

    for (int i = 0; i < maxIter; ++i) {
        a = a * a + c;
        if (a.magnitude2() > 1000.0f) return 0; // 발산
    }
    return 1; // 내부
}

// 간단한 PPM 저장 (binary P6)
static void savePPM(const char *filepath, const unsigned char *rgb, int width, int height) {
    std::FILE *fp = std::fopen(filepath, "wb");
    if (!fp) {
        std::perror("fopen");
        std::exit(EXIT_FAILURE);
    }
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

    // Host RGB 버퍼
    size_t numBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    unsigned char *rgb = static_cast<unsigned char *>(std::malloc(numBytes));
    if (!rgb) {
        std::fprintf(stderr, "malloc failed for rgb (%zu bytes)\n", numBytes);
        std::exit(EXIT_FAILURE);
    }

    // 2중 for 루프: 픽셀 단위로 Julia 판정 수행 (CPU 직렬 버전)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int inside = juliaAt(x, y, width, height, JULIA_C_RE, JULIA_C_IM, MAX_ITER);
            int offset = (y * width + x) * 3;
            rgb[offset + 0] = static_cast<unsigned char>(255 * inside);
            rgb[offset + 1] = 0;
            rgb[offset + 2] = 0;
        }
    }

    savePPM("julia_cpu.ppm", rgb, width, height);
    std::printf("[OK] PPM saved: julia_cpu.ppm (%dx%d)\n", width, height);

    std::free(rgb);
    return 0;
} 