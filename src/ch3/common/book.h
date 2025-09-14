#ifndef BOOK_H_
#define BOOK_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 간단한 에러 체크 유틸리티
static inline void __handleCudaError(cudaError_t error,
                                     const char *file,
                                     int line,
                                     const char *expr) {
    if (error != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA Error at %s:%d\n  expr: %s\n  code: %d, reason: %s\n",
                     file, line, expr, static_cast<int>(error), cudaGetErrorString(error));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err_expr) (__handleCudaError((err_expr), __FILE__, __LINE__, #err_expr))

// 커널 런치 후 즉시 마지막 에러를 확인
#define CHECK_LAST_CUDA_ERROR() HANDLE_ERROR(cudaGetLastError())

// 커널 완료를 기다리고 마지막 에러를 확인(디버깅에 유용)
#define SYNC_AND_CHECK() do { \
    HANDLE_ERROR(cudaDeviceSynchronize()); \
    CHECK_LAST_CUDA_ERROR(); \
} while (0)

#endif // BOOK_H_ 