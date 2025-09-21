NVCC ?= nvcc
NVCCFLAGS ?= -O2

BIN_DIR := ./bin
SRC_DIR := .

VECTOR_ADD_THREADS    := $(BIN_DIR)/vector_add_threads
VECTOR_ADD_1D_GRID    := $(BIN_DIR)/vector_add_1d_grid
VECTOR_ADD_STRIDE     := $(BIN_DIR)/vector_add_stride
RIPPLE_PPM            := $(BIN_DIR)/ripple_ppm
DOT_PRODUCT_SHARED    := $(BIN_DIR)/dot_product_shared

.PHONY: all clean \
        run_vector_add_threads run_vector_add_1d_grid run_vector_add_stride \
        run_ripple_ppm run_dot_product_shared

all: $(VECTOR_ADD_THREADS) $(VECTOR_ADD_1D_GRID) $(VECTOR_ADD_STRIDE) $(RIPPLE_PPM) $(DOT_PRODUCT_SHARED)
	@echo "Build complete. Binaries are in $(BIN_DIR)"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(VECTOR_ADD_THREADS): $(SRC_DIR)/vector_add_threads.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(VECTOR_ADD_1D_GRID): $(SRC_DIR)/vector_add_1d_grid.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(VECTOR_ADD_STRIDE): $(SRC_DIR)/vector_add_stride.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(RIPPLE_PPM): $(SRC_DIR)/ripple_ppm.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(DOT_PRODUCT_SHARED): $(SRC_DIR)/dot_product_shared.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run_vector_add_threads: $(VECTOR_ADD_THREADS)
	$<

run_vector_add_1d_grid: $(VECTOR_ADD_1D_GRID)
	$<

run_vector_add_stride: $(VECTOR_ADD_STRIDE)
	$<

run_ripple_ppm: $(RIPPLE_PPM)
	$<
	@echo "\nGenerated image: ./ripple.ppm"
	@echo "You can open it with an image viewer that supports PPM."

run_dot_product_shared: $(DOT_PRODUCT_SHARED)
	$<

clean:
	rm -rf $(BIN_DIR) *.ppm 