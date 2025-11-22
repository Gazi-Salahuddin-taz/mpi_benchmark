# src/Makefile - Core Library Build

# Configuration from top-level
BUILD_TYPE ?= release
BUILD_DIR ?= ../build/src
INSTALL_DIR ?= ../install

# Source files
CORE_SRCS = $(wildcard core/*.cpp)
ALGORITHMS_SRCS = $(wildcard algorithms/*.cpp)
UTILS_SRCS = $(wildcard utils/*.cpp)
ANALYSIS_SRCS = $(wildcard analysis/*.cpp)

ALL_SRCS = $(CORE_SRCS) $(ALGORITHMS_SRCS) $(UTILS_SRCS) $(ANALYSIS_SRCS)
ALL_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(ALL_SRCS))
ALL_DEPS = $(patsubst %.cpp,$(BUILD_DIR)/%.d,$(ALL_SRCS))

# Compiler and flags
CXX = mpicxx
CXX_STD = c++17
WARN_FLAGS = -Wall -Wextra -Wpedantic
BASE_FLAGS = -std=$(CXX_STD) $(WARN_FLAGS) -fopenmp -I.

# Build type specific flags
ifeq ($(BUILD_TYPE), release)
	 CXXFLAGS = $(BASE_FLAGS) -O3 -march=native -DNDEBUG -ffast-math
else ifeq ($(BUILD_TYPE), debug)
	 CXXFLAGS = $(BASE_FLAGS) -O0 -g -DDEBUG -fno-omit-frame-pointer
else ifeq ($(BUILD_TYPE), pgo)
	 CXXFLAGS = $(BASE_FLAGS) -O3 -march=native -fprofile-generate
	 LDFLAGS = -fprofile-generate
else
	 CXXFLAGS = $(BASE_FLAGS) -O2
endif

# Library targets
LIB_NAME = libtopology_aware_mpi.a
LIB_TARGET = $(BUILD_DIR)/$(LIB_NAME)
INSTALL_LIB = $(INSTALL_DIR)/lib/$(LIB_NAME)

# Main target
all: $(LIB_TARGET)

# Create build directories
$(BUILD_DIR)/%.o: %.cpp
	   @mkdir -p $(dir $@)
	   $(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Build library
$(LIB_TARGET): $(ALL_OBJS)
	   @echo "Building library: $@"
	   @mkdir -p $(dir $@)
	   ar rcs $@ $^
	   @echo "Library built successfully: $@"

# Install headers and library
install: $(LIB_TARGET)
	   @echo "Installing core library to $(INSTALL_DIR)"
	   @mkdir -p $(INSTALL_DIR)/include
	   @mkdir -p $(INSTALL_DIR)/lib

	   # Install headers
	   find . -name '*.h' -exec cp --parents {} $(INSTALL_DIR)/include/ \;

	   # Install library
	   cp $(LIB_TARGET) $(INSTALL_LIB)

	   @echo "Installation completed: $(INSTALL_LIB)"

# Include dependencies
-include $(ALL_DEPS)

# Clean
clean:
	   rm -rf $(BUILD_DIR)
	   @echo "Core library build cleaned"

.PHONY: all install clean