NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17
INCLUDES := -Icommon

SRC_DIR := .
BIN_DIR := bin

SOURCES := \
    $(SRC_DIR)/01_hello_host_only.cu \
    $(SRC_DIR)/02_empty_kernel_launch.cu \
    $(SRC_DIR)/03_add_with_device_memory.cu \
    $(SRC_DIR)/04_query_devices.cu \
    $(SRC_DIR)/05_choose_device.cu

BINS := $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(SOURCES))

.PHONY: all clean run-%

all: $(BIN_DIR) $(BINS)
	@echo "Built: $(notdir $(BINS))"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu common/book.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@.out $<

run-%: $(BIN_DIR)/%
	$< | cat

clean:
	rm -rf $(BIN_DIR) 