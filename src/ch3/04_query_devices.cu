#include <cstdio>
#include <cstring>
#include "common/book.h"

// 3.4 디바이스 조회 — 시스템 내 CUDA 디바이스 정보를 출력
// 비고: 일부 필드는 플랫폼/툴킷에 따라 자료형이 다를 수 있어 형변환을 통해 안전하게 출력합니다.

// --- General Information for device 0 ---
// Name: NVIDIA L40S
// Compute capability: 8.9
// Clock rate (kHz): 2520000
// Device copy overlap: Enabled
// Kernel execution timeout: Disabled
// --- Memory Information ---
// Total global mem: 47810936832 bytes
// Total constant mem: 65536 bytes
// Max mem pitch: 2147483647 bytes
// Texture Alignment: 512 bytes
// --- MP Information ---
// Multiprocessor count: 142
// Shared mem per block: 49152 bytes
// Registers per block: 65536
// Threads in warp: 32
// Max threads per block: 1024
// Max thread dimensions: (1024, 1024, 64)
// Max grid dimensions: (2147483647, 65535, 65535)

int main() {
    int device_count = 0;
    HANDLE_ERROR(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::printf("No CUDA devices found.\n");
        return 0;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));

        std::printf("--- General Information for device %d ---\n", i);
        std::printf("Name: %s\n", prop.name);
        std::printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        std::printf("Clock rate (kHz): %d\n", prop.clockRate);
        std::printf("Device copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
        std::printf("Kernel execution timeout: %s\n",
                    prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        std::printf("--- Memory Information ---\n");
        std::printf("Total global mem: %llu bytes\n", static_cast<unsigned long long>(prop.totalGlobalMem));
        std::printf("Total constant mem: %llu bytes\n", static_cast<unsigned long long>(prop.totalConstMem));
        std::printf("Max mem pitch: %llu bytes\n", static_cast<unsigned long long>(prop.memPitch));
        std::printf("Texture Alignment: %llu bytes\n", static_cast<unsigned long long>(prop.textureAlignment));

        std::printf("--- MP Information ---\n");
        std::printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        std::printf("Shared mem per block: %llu bytes\n",
                    static_cast<unsigned long long>(prop.sharedMemPerBlock));
        std::printf("Registers per block: %d\n", prop.regsPerBlock);
        std::printf("Threads in warp: %d\n", prop.warpSize);
        std::printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        std::printf("Max thread dimensions: (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        std::printf("Max grid dimensions: (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        std::printf("\n");
    }

    return 0;
}