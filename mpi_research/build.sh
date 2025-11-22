#!/bin/bash

# Top-level build script for MPI Research Project
# Orchestrates compilation of all components in correct order

set -e  # Exit on error

# Configuration
PROJECT_NAME="Topology-Aware MPI Research"
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Build configuration
BUILD_TYPE="${BUILD_TYPE:-release}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"
NS3_HOME="${NS3_HOME:-$HOME/wangyl/ns-allinone-3.37/ns-3.37}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build the complete MPI Research project"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  -t, --type TYPE         Build type (release, debug, pgo)"
    echo "  -j, --jobs N            Number of parallel jobs"
    echo "  --ns3-home DIR          NS-3 installation directory"
    echo "  --no-ns3                Skip NS-3 integration build"
    echo "  --no-benchmarks         Skip benchmarks build"
    echo "  --configure-only        Only configure, don't build"
    echo "  --clean                 Clean before building"
    echo "  --list-targets          List available build targets"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 -t release -j8              # Release build with 8 jobs"
    echo "  $0 -t debug --no-ns3          # Debug build without NS-3"
    echo "  $0 --clean -t release         # Clean and release build"
}

# Validate environment
validate_environment() {
    log_info "Validating build environment..."

    # Check compiler
    if ! command -v mpicxx &> /dev/null; then
        log_error "MPI compiler (mpicxx) not found. Please install MPI."
        exit 1
    fi

    # Check CMake (for potential future use)
    if ! command -v cmake &> /dev/null; then
        log_warning "CMake not found. Using Makefile build system."
    fi

    log_success "Build environment validated"
}

# Clean project
clean_project() {
    log_info "Cleaning project..."
    make clean
    log_success "Project cleaned"
}

# Build core library
build_core() {
    log_info "Building core library..."

    if make src -j "$PARALLEL_JOBS" BUILD_TYPE="$BUILD_TYPE"; then
        log_success "Core library built successfully"
    else
        log_error "Failed to build core library"
        exit 1
    fi
}

# Build benchmarks
build_benchmarks() {
    if [ "$SKIP_BENCHMARKS" = true ]; then
        log_info "Skipping benchmarks build"
        return 0
    fi

    log_info "Building benchmarks..."

    if make benchmarks -j "$PARALLEL_JOBS" BUILD_TYPE="$BUILD_TYPE"; then
        log_success "Benchmarks built successfully"
    else
        log_error "Failed to build benchmarks"
        exit 1
    fi
}

# Build NS-3 integration
build_ns3() {
    if [ "$SKIP_NS3" = true ]; then
        log_info "Skipping NS-3 integration build"
        return 0
    fi

    log_info "Building NS-3 integration..."

    if make ns3 -j "$PARALLEL_JOBS" NS3_HOME="$NS3_HOME" BUILD_TYPE="$BUILD_TYPE"; then
        log_success "NS-3 integration built successfully"
    else
        log_warning "NS-3 integration build failed or skipped"
        log_info "To build without NS-3, use --no-ns3 flag"
    fi
}

# Run tests
run_tests() {
    if [ "$SKIP_BENCHMARKS" = true ]; then
        log_info "Skipping tests (benchmarks not built)"
        return 0
    fi

    log_info "Running tests..."

    if make test; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        exit 1
    fi
}

# Main build function
main_build() {
    log_info "Starting $PROJECT_NAME build"
    log_info "Build Type: $BUILD_TYPE"
    log_info "Parallel Jobs: $PARALLEL_JOBS"
    log_info "Project Root: $PROJECT_ROOT"

    # Validate environment
    validate_environment

    # Clean if requested
    if [ "$CLEAN_FIRST" = true ]; then
        clean_project
    fi

    # Build components in correct order
    build_core
    build_benchmarks
    build_ns3

    # Run tests
    run_tests

    log_success "=== BUILD COMPLETED SUCCESSFULLY ==="
    log_info "Project built in $BUILD_TYPE mode"
    log_info "Installation directory: $PROJECT_ROOT/install"
    log_info "Build directory: $PROJECT_ROOT/build"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -j|--jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --ns3-home)
                NS3_HOME="$2"
                shift 2
                ;;
            --no-ns3)
                SKIP_NS3=true
                shift
                ;;
            --no-benchmarks)
                SKIP_BENCHMARKS=true
                shift
                ;;
            --configure-only)
                CONFIGURE_ONLY=true
                shift
                ;;
            --clean)
                CLEAN_FIRST=true
                shift
                ;;
            --list-targets)
                echo "Available build targets:"
                echo "  src         - Core MPI optimization library"
                echo "  benchmarks  - Performance benchmarks and tests"
                echo "  ns3         - NS-3 network simulations"
                echo "  all         - Complete project (default)"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate build type
validate_build_type() {
    case $BUILD_TYPE in
        release|debug|pgo)
            # Valid build type
            ;;
        *)
            log_error "Invalid build type: $BUILD_TYPE"
            log_info "Valid types: release, debug, pgo"
            exit 1
            ;;
    esac
}

# Main function
main() {
    parse_arguments "$@"
    validate_build_type

    if [ "$CONFIGURE_ONLY" = true ]; then
        log_info "Configuration only - not building"
        make config BUILD_TYPE="$BUILD_TYPE"
    else
        main_build
    fi
}

# Run main function
main "$@"