# Top-Level Makefile for MPI Research Project
# Orchestrates compilation of src/, benchmarks/, and ns3_integration/

# Configuration
PROJECT_NAME = topology_aware_mpi
VERSION = 1.0.0

# Directories
SRC_DIR = src
BENCHMARKS_DIR = benchmarks
NS3_DIR = ns3_integration
BUILD_DIR = build
INSTALL_DIR = install
RESULTS_DIR = results

# Compiler and flags
CXX = mpicxx
CXX_STD = c++17
OPT_FLAGS = -O3 -march=native -DNDEBUG
DEBUG_FLAGS = -O0 -g -DDEBUG
WARN_FLAGS = -Wall -Wextra -Wpedantic
CXXFLAGS = -std=$(CXX_STD) $(WARN_FLAGS) -fopenmp

# Build type: release or debug
BUILD_TYPE = release

# NS-3 configuration
NS3_HOME ?= $(HOME)/wangyl/ns-allinone-3.37/ns-3.37
NS3_BUILD_TYPE = release

# Parallel build
JOBS = $(shell nproc)

# Default target
.DEFAULT_GOAL := all

# Show build configuration
config:
	@echo "=== $(PROJECT_NAME) Build Configuration ==="
	@echo "Project: $(PROJECT_NAME) v$(VERSION)"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo "Compiler: $(CXX)"
	@echo "C++ Standard: $(CXX_STD)"
	@echo "Install Directory: $(INSTALL_DIR)"
	@echo "NS-3 Directory: $(NS3_HOME)"
	@echo "Parallel Jobs: $(JOBS)"
	@echo "========================================"

# Create directory structure
directories:
	@mkdir -p $(BUILD_DIR)/src
	@mkdir -p $(BUILD_DIR)/benchmarks
	@mkdir -p $(INSTALL_DIR)/lib
	@mkdir -p $(INSTALL_DIR)/include
	@mkdir -p $(INSTALL_DIR)/bin
	@mkdir -p $(RESULTS_DIR)

# Build core library
src: directories
	@echo "=== Building Core Library ==="
	$(MAKE) -C $(SRC_DIR) BUILD_TYPE=$(BUILD_TYPE) \
		BUILD_DIR=../$(BUILD_DIR)/src \
		INSTALL_DIR=../$(INSTALL_DIR) \
		-j $(JOBS) install

# Build benchmarks (depends on core library)
benchmarks: src
	@echo "=== Building Benchmarks ==="
	$(MAKE) -C $(BENCHMARKS_DIR) BUILD_TYPE=$(BUILD_TYPE) \
		BUILD_DIR=../$(BUILD_DIR)/benchmarks \
		INSTALL_DIR=../$(INSTALL_DIR) \
		-j $(JOBS)

# Build NS-3 integration (separate build system)
ns3: src
	@echo "=== Building NS-3 Integration ==="
	$(MAKE) -C $(NS3_DIR) NS3_HOME=$(NS3_HOME) \
		BUILD_TYPE=$(NS3_BUILD_TYPE) \
		INSTALL_DIR=../$(INSTALL_DIR) \
		-j $(JOBS)

# Build everything in correct order
all: src benchmarks ns3

# Development build with debug symbols
debug: BUILD_TYPE = debug
debug: all

# Profile-guided optimization build
pgo: BUILD_TYPE = pgo
pgo: all

# Installation
install: all
	@echo "=== Installing $(PROJECT_NAME) ==="
	@cp -r $(INSTALL_DIR)/* /usr/local/ 2>/dev/null || \
		echo "Installation to /usr/local/ requires sudo privileges"
	@echo "Installation completed"

# Run tests
test: benchmarks
	@echo "=== Running Tests ==="
	@cd $(BUILD_DIR)/benchmarks && \
		mpirun -np 4 ./validation/correctness_tests && \
		mpirun -np 2 ./validation/unit_tests

bench: benchmarks
	@echo "=== Running Benchmarks ==="
	@mkdir -p $(RESULTS_DIR)/benchmarks
	@cd $(BUILD_DIR)/benchmarks && \
		mpirun -np 8 ./micro/broadcast_benchmark && \
		mpirun -np 8 ./micro/collective_benchmark

scalability: benchmarks
	@echo "=== Running Scalability Analysis ==="
	@mkdir -p $(RESULTS_DIR)/scalability
	@cd $(BUILD_DIR)/benchmarks && \
		mpirun -np 16 ./scalability/scalability_analysis && \
		mpirun -np 8 ./scalability/weak_strong_scaling


# Run NS-3 simulations
simulate: ns3
	@echo "=== Running NS-3 Simulations ==="
	@mkdir -p $(RESULTS_DIR)/ns3_simulations
	@cd $(NS3_DIR) && \
		./scripts/run_ns3_simulations.sh -t fat_tree -o ../$(RESULTS_DIR)/ns3_simulations/fat_tree && \
		./scripts/run_ns3_simulations.sh -t torus_2d -o ../$(RESULTS_DIR)/ns3_simulations/torus_2d

# Analyze all results
analyze: bench scalability simulate
	@echo "=== Analyzing Results ==="
	@cd $(NS3_DIR) && \
		./scripts/analyze_ns3_results.py ../$(RESULTS_DIR)/ns3_simulations --detailed --output ../$(RESULTS_DIR)/analysis
	@echo "Results analysis completed: $(RESULTS_DIR)/analysis/"

# Clean build artifacts
clean:
	@echo "=== Cleaning Build Artifacts ==="
	$(MAKE) -C $(SRC_DIR) clean
	$(MAKE) -C $(BENCHMARKS_DIR) clean
	$(MAKE) -C $(NS3_DIR) clean
	@rm -rf $(BUILD_DIR)
	@rm -rf $(INSTALL_DIR)
	@echo "Clean completed"

# Deep clean (including results)
distclean: clean
	@echo "=== Deep Cleaning ==="
	@rm -rf $(RESULTS_DIR)
	@rm -f *.log
	@rm -f *.out
	@echo "Distclean completed"

# Generate documentation
docs:
	@echo "=== Generating Documentation ==="
	@doxygen Doxyfile 2>/dev/null || echo "Doxygen not found, skipping documentation"
	@echo "Documentation generated in docs/"

# Package release
package: all docs
	@echo "=== Creating Release Package ==="
	@mkdir -p dist/$(PROJECT_NAME)-$(VERSION)
	@cp -r $(INSTALL_DIR)/* dist/$(PROJECT_NAME)-$(VERSION)/
	@cp -r docs/ dist/$(PROJECT_NAME)-$(VERSION)/
	@cp README.md LICENSE dist/$(PROJECT_NAME)-$(VERSION)/
	@tar -czf dist/$(PROJECT_NAME)-$(VERSION).tar.gz -C dist $(PROJECT_NAME)-$(VERSION)
	@echo "Package created: dist/$(PROJECT_NAME)-$(VERSION).tar.gz"

# Show help
help:
	@echo "=== $(PROJECT_NAME) Build System ==="
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  config       - Show build configuration"
	@echo "  src          - Build core library only"
	@echo "  benchmarks   - Build benchmarks (requires src)"
	@echo "  ns3          - Build NS-3 integration (requires src)"
	@echo "  debug        - Build with debug symbols"
	@echo "  pgo          - Build with profile-guided optimization"
	@echo "  install      - Install to system"
	@echo "  test         - Run correctness tests"
	@echo "  bench        - Run performance benchmarks"
	@echo "  scalability  - Run scalability analysis"
	@echo "  simulate     - Run NS-3 simulations"
	@echo "  analyze      - Run complete analysis pipeline"
	@echo "  clean        - Clean build artifacts"
	@echo "  distclean    - Clean everything including results"
	@echo "  docs         - Generate documentation"
	@echo "  package      - Create release package"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make all -j8          # Build everything with 8 jobs"
	@echo "  make debug            # Build with debug symbols"
	@echo "  make bench && make analyze  # Run benchmarks and analyze"

# Phony targets
.PHONY: all config src benchmarks ns3 debug pgo install test bench scalability \
        simulate analyze clean distclean docs package help directories