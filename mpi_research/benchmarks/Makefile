# benchmarks/Makefile - Benchmark Build

# Configuration from top-level
BUILD_TYPE ?= release
BUILD_DIR ?= ../build/benchmarks
INSTALL_DIR ?= ../install

# Source files
MICRO_SRCS = $(wildcard micro/*.cpp)
SCALABILITY_SRCS = $(wildcard scalability/*.cpp)
VALIDATION_SRCS = $(wildcard validation/*.cpp)

VALIDATION_SRCS = $(wildcard validation/*.cpp)
ALL_SRCS = $(MICRO_SRCS) $(SCALABILITY_SRCS) $(VALIDATION_SRCS)

ALL_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(ALL_SRCS))

BENCHMARK_EXECS = $(patsubst %.cpp,$(BUILD_DIR)/%,$(MICRO_SRCS)) \
                  $(patsubst %.cpp,$(BUILD_DIR)/%,$(SCALABILITY_SRCS)) \
                  $(patsubst %.cpp,$(BUILD_DIR)/%,$(VALIDATION_SRCS))

# Compiler and flags
CXX = mpicxx
CXX_STD = c++17
WARN_FLAGS = -Wall -Wextra -Wpedantic

# Include paths
INC_DIRS = -I../src -I$(INSTALL_DIR)/include
BASE_FLAGS = -std=$(CXX_STD) $(WARN_FLAGS) -fopenmp $(INC_DIRS)

# Build type specific flags
ifeq ($(BUILD_TYPE), release)
	 CXXFLAGS = $(BASE_FLAGS) -O3 -march=native -DNDEBUG -ffast-math
else ifeq ($(BUILD_TYPE), debug)
	 CXXFLAGS = $(BASE_FLAGS) -O0 -g -DDEBUG -fno-omit-frame-pointer
else ifeq ($(BUILD_TYPE), pgo)
	 CXXFLAGS = $(BASE_FLAGS) -O3 -march=native -fprofile-use
	 LDFLAGS = -fprofile-use
else
	 CXXFLAGS = $(BASE_FLAGS) -O2
endif

# Library dependencies
LIB_DIR = -L$(INSTALL_DIR)/lib
LIBS = -ltopology_aware_mpi -lm

# Main target
all: $(BENCHMARK_EXECS)

# Create build directories and compile
$(BUILD_DIR)/%.o: %.cpp
	   @mkdir -p $(dir $@)
	   $(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Link executables
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	   @mkdir -p $(dir $@)
	   $(CXX) $(CXXFLAGS) -o $@ $< $(LIB_DIR) $(LIBS)
	   @echo "Built: $@"

# Specific executable dependencies
$(BUILD_DIR)/micro/broadcast_benchmark: $(BUILD_DIR)/micro/broadcast_benchmark.o
$(BUILD_DIR)/micro/collective_benchmark: $(BUILD_DIR)/micro/collective_benchmark.o
$(BUILD_DIR)/micro/reduce_benchmark: $(BUILD_DIR)/micro/reduce_benchmark.o
$(BUILD_DIR)/scalability/scalability_analysis: $(BUILD_DIR)/scalability/scalability_analysis.o
$(BUILD_DIR)/scalability/weak_strong_scaling: $(BUILD_DIR)/scalability/weak_strong_scaling.o
$(BUILD_DIR)/validation/correctness_tests: $(BUILD_DIR)/validation/correctness_tests.o
$(BUILD_DIR)/validation/unit_tests: $(BUILD_DIR)/validation/unit_tests.o

# Install benchmarks
install: all
	   @echo "Installing benchmarks to $(INSTALL_DIR)/bin"
	   @mkdir -p $(INSTALL_DIR)/bin
	   cp $(BENCHMARK_EXECS) $(INSTALL_DIR)/bin/
	   @echo "Benchmarks installed"

# Include dependencies
-include $(ALL_DEPS)

# Clean
clean:
	   rm -rf $(BUILD_DIR)
	   @echo "Benchmarks build cleaned"

.PHONY: all install clean