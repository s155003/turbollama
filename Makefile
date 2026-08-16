# turbollama build.
#
# The scalar baseline is deliberately compiled with plain -O3 and NO -march
# override: that is exactly what a stock `make runq` of llama2.c produces, so
# the speedups we report are against a real default build, not a hobbled one.
# Only the SIMD translation units get -march, and they are entered solely after
# a runtime HWCAP check -- so a single binary stays loadable on Armv8.0.

CC      ?= gcc
ARCH    := $(shell uname -m)
BUILD   := build

CFLAGS  ?= -O3 -fno-math-errno -funroll-loops -Wall -Wno-unused-function
INCS    := -Isrc
LDLIBS  := -lm

ifeq ($(ARCH),aarch64)
  DOT_FLAGS  := -march=armv8.2-a+dotprod
  I8MM_FLAGS := -march=armv8.2-a+i8mm
else
  DOT_FLAGS  :=
  I8MM_FLAGS :=
endif

# make OMP=1 to enable the OpenMP parallel-for in every kernel
ifeq ($(OMP),1)
  CFLAGS += -fopenmp
  LDLIBS += -fopenmp
endif

CORE_SRC := src/kernels_scalar.c src/kernels_dot.c src/kernels_i8mm.c src/dispatch.c
CORE_OBJ := $(CORE_SRC:%.c=$(BUILD)/%.o)

.PHONY: all bench runq test clean info
all: bench runq

$(BUILD)/src/kernels_dot.o: src/kernels_dot.c src/kernels.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DOT_FLAGS) $(INCS) -c $< -o $@

$(BUILD)/src/kernels_i8mm.o: src/kernels_i8mm.c src/kernels.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(I8MM_FLAGS) $(INCS) -c $< -o $@

$(BUILD)/%.o: %.c src/kernels.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@

bench: $(BUILD)/bench_kernel
$(BUILD)/bench_kernel: $(CORE_OBJ) $(BUILD)/bench/bench_kernel.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

runq: $(BUILD)/runq_tl
$(BUILD)/runq_tl: $(CORE_OBJ) $(BUILD)/src/runq_tl.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: bench
	./$(BUILD)/bench_kernel

info:
	@echo "arch       : $(ARCH)"
	@echo "cc         : $(shell $(CC) --version | head -1)"
	@echo "dot flags  : $(DOT_FLAGS)"
	@echo "i8mm flags : $(I8MM_FLAGS)"

clean:
	rm -rf $(BUILD)
