# 간단한 CUDA 예제 빌드용 Makefile (src/ch4)

NVCC ?= nvcc
NVCCFLAGS ?= -O2

BIN_DIR := ./bin
SRC_DIR := .

VECTOR_ADD      := $(BIN_DIR)/vector_add
JULIA_PPM       := $(BIN_DIR)/julia_ppm
VECTOR_ADD_CPU  := $(BIN_DIR)/vector_add_cpu
JULIA_CPU_PPM   := $(BIN_DIR)/julia_cpu_ppm
VIS_XY_PPM      := $(BIN_DIR)/visualize_xy_ppm

.PHONY: all clean run_vector_add run_julia_ppm run_vector_add_cpu run_julia_cpu_ppm run_visualize_xy

all: $(VECTOR_ADD) $(JULIA_PPM) $(VECTOR_ADD_CPU) $(JULIA_CPU_PPM) $(VIS_XY_PPM)
	@echo "Build complete. Binaries are in $(BIN_DIR)"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(VECTOR_ADD): $(SRC_DIR)/vector_add.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(JULIA_PPM): $(SRC_DIR)/julia_ppm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(VECTOR_ADD_CPU): $(SRC_DIR)/vector_add_cpu.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(JULIA_CPU_PPM): $(SRC_DIR)/julia_cpu_ppm.cpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(VIS_XY_PPM): $(SRC_DIR)/visualize_xy_ppm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run_vector_add: $(VECTOR_ADD)
	$<

run_julia_ppm: $(JULIA_PPM)
	$<
	@echo "\nGenerated image: ./julia.ppm"
	@echo "You can open it with an image viewer that supports PPM."

run_vector_add_cpu: $(VECTOR_ADD_CPU)
	$<

run_julia_cpu_ppm: $(JULIA_CPU_PPM)
	$<
	@echo "\nGenerated image: ./julia_cpu.ppm"
	@echo "You can open it with an image viewer that supports PPM."

run_visualize_xy: $(VIS_XY_PPM)
	$<
	@echo "\nGenerated image: ./xy_mapping.ppm"
	@echo "It visualizes global (x,y) and block borders with labels."

clean:
	rm -rf $(BIN_DIR) *.ppm 