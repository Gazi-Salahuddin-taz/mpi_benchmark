#!/bin/bash
#!/bin/bash
#!/bin/bash

# Advanced Dependency Installation Script for MPI Research
# Installs all required dependencies for the project

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/install.log"
BUILD_DIR="$PROJECT_ROOT/build"
INSTALL_DIR="$PROJECT_ROOT/install"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
	   echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
	   echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
	   echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
	   echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Detect operating system
detect_os() {
	   if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	       echo "linux"
	   elif [[ "$OSTYPE" == "darwin"* ]]; then
	       echo "macos"
	   elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
	       echo "windows"
	   else
	       echo "unknown"
	   fi
}

# Detect package manager
detect_package_manager() {
	   if command -v apt-get &> /dev/null; then
	       echo "apt"
	   elif command -v yum &> /dev/null; then
	       echo "yum"
	   elif command -v dnf &> /dev/null; then
	       echo "dnf"
	   elif command -v pacman &> /dev/null; then
	       echo "pacman"
	   elif command -v brew &> /dev/null; then
	       echo "brew"
	   else
	       echo "unknown"
	   fi
}

# Check if command exists
command_exists() {
	   command -v "$1" &> /dev/null
}

# Install system dependencies
install_system_dependencies() {
	   local os=$(detect_os)
	   local pkg_manager=$(detect_package_manager)
	   
	   log_info "Installing system dependencies for $os using $pkg_manager..."
	   
	   case "$pkg_manager" in
	       apt)
	           sudo apt-get update
	           sudo apt-get install -y \
	               build-essential \
	               cmake \
	               git \
	               wget \
	               curl \
	               python3 \
	               python3-pip \
	               python3-venv \
	               openmpi-bin \
	               libopenmpi-dev \
	               gnuplot \
	               doxygen \
	               graphviz \
	               libboost-all-dev \
	               libeigen3-dev
	           ;;
	       yum|dnf)
	           sudo $pkg_manager update -y
	           sudo $pkg_manager install -y \
	               gcc-c++ \
	               cmake \
	               git \
	               wget \
	               curl \
	               python3 \
	               python3-pip \
	               openmpi \
	               openmpi-devel \
	               gnuplot \
	               doxygen \
	               graphviz \
	               boost-devel \
	               eigen3-devel
	           ;;
	       pacman)
	           sudo pacman -Syu --noconfirm
	           sudo pacman -S --noconfirm \
	               base-devel \
	               cmake \
	               git \
	               wget \
	               curl \
	               python \
	               python-pip \
	               openmpi \
	               gnuplot \
	               doxygen \
	               graphviz \
	               boost \
	               eigen
	           ;;
	       brew)
	           brew update
	           brew install \
	               cmake \
	               git \
	               wget \
	               curl \
	               python \
	               open-mpi \
	               gnuplot \
	               doxygen \
	               graphviz \
	               boost \
	               eigen
	           ;;
	       *)
	           log_warning "Unknown package manager. Please install dependencies manually."
	           return 1
	           ;;
	   esac
	   
	   log_success "System dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
	   log_info "Installing Python dependencies..."
	   
	   # Create virtual environment
	   if [ ! -d "$PROJECT_ROOT/venv" ]; then
	       python3 -m venv "$PROJECT_ROOT/venv"
	   fi
	   
	   # Activate virtual environment
	   source "$PROJECT_ROOT/venv/bin/activate"
	   
	   # Upgrade pip
	   pip install --upgrade pip
	   
	   # Install required packages
	   pip install \
	       numpy \
	       scipy \
	       pandas \
	       matplotlib \
	       seaborn \
	       jupyter \
	       plotly \
	       scienceplots \
	       statsmodels \
	       scikit-learn \
	       ipykernel
	   
	   # Install development packages
	   pip install \
	       black \
	       flake8 \
	       mypy \
	       pytest \
	       pytest-cov
	   
	   log_success "Python dependencies installed"
}

# Install NS-3 dependencies
install_ns3_dependencies() {
	   log_info "Installing NS-3 dependencies..."
	   
	   local os=$(detect_os)
	   local pkg_manager=$(detect_package_manager)
	   
	   case "$pkg_manager" in
	       apt)
	           sudo apt-get install -y \
	               mercurial \
	               qt5-default \
	               qt5-qmake \
	               qtbase5-dev \
	               qttools5-dev \
	               qttools5-dev-tools \
	               qtdeclarative5-dev \
	               libxml2-dev \
	               libgtk-3-dev \
	               libgsl-dev \
	               libboost-all-dev \
	               libsqlite3-dev \
	               libgdbm-dev
	           ;;
	       yum|dnf)
	           sudo $pkg_manager install -y \
	               mercurial \
	               qt5-qtbase-devel \
	               qt5-qttools-devel \
	               libxml2-devel \
	               gtk3-devel \
	               gsl-devel \
	               boost-devel \
	               sqlite-devel \
	               gdbm-devel
	           ;;
	       pacman)
	           sudo pacman -S --noconfirm \
	               mercurial \
	               qt5-base \
	               qt5-tools \
	               libxml2 \
	               gtk3 \
	               gsl \
	               boost \
	               sqlite \
	               gdbm
	           ;;
	       brew)
	           brew install \
	               mercurial \
	               qt \
	               gsl \
	               boost \
	               sqlite \
	               gdbm
	           ;;
	   esac
	   
	   log_success "NS-3 dependencies installed"
}

# Download and build NS-3
install_ns3() {
	   local ns3_version="ns-3.37"
	   local ns3_dir="$PROJECT_ROOT/ns-3-allinone"
	   
	   log_info "Installing NS-3..."
	   
	   if [ -d "$ns3_dir" ]; then
	       log_info "NS-3 directory exists, skipping download"
	   else
	       # Download NS-3
	       wget "https://www.nsnam.org/releases/${ns3_version}.tar.bz2" -O "/tmp/${ns3_version}.tar.bz2"
	       
	       # Extract
	       tar -xjf "/tmp/${ns3_version}.tar.bz2" -C "$PROJECT_ROOT"
	       mv "$PROJECT_ROOT/$ns3_version" "$ns3_dir"
	       
	       # Cleanup
	       rm "/tmp/${ns3_version}.tar.bz2"
	   fi
	   
	   # Build NS-3
	   cd "$ns3_dir"
	   ./build.py --enable-examples --enable-tests
	   
	   log_success "NS-3 installed successfully"
}

# Verify installations
verify_installations() {
	   log_info "Verifying installations..."
	   
	   local errors=0
	   
	   # Check MPI
	   if command_exists mpicc && command_exists mpirun; then
	       log_success "MPI installation verified"
	   else
	       log_error "MPI installation failed"
	       ((errors++))
	   fi
	   
	   # Check Python
	   if command_exists python3; then
	       python3 --version >> "$LOG_FILE"
	       log_success "Python installation verified"
	   else
	       log_error "Python installation failed"
	       ((errors++))
	   fi
	   
	   # Check CMake
	   if command_exists cmake; then
	       cmake --version >> "$LOG_FILE"
	       log_success "CMake installation verified"
	   else
	       log_error "CMake installation failed"
	       ((errors++))
	   fi
	   
	   # Check Git
	   if command_exists git; then
	       git --version >> "$LOG_FILE"
	       log_success "Git installation verified"
	   else
	       log_error "Git installation failed"
	       ((errors++))
	   fi
	   
	   return $errors
}

# Setup environment variables
setup_environment() {
	   log_info "Setting up environment variables..."
	   
	   local env_file="$PROJECT_ROOT/.env"
	   
	   cat > "$env_file" << EOF
# MPI Research Project Environment Variables
export PROJECT_ROOT="$PROJECT_ROOT"
export BUILD_DIR="$BUILD_DIR"
export INSTALL_DIR="$INSTALL_DIR"
export NS3_HOME="$PROJECT_ROOT/ns-3-allinone/ns-3.37"
export PATH="\$INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PYTHONPATH="\$PROJECT_ROOT/scripts:\$PYTHONPATH"

# Python virtual environment
export VIRTUAL_ENV="$PROJECT_ROOT/venv"
export PATH="\$VIRTUAL_ENV/bin:\$PATH"

# MPI configuration
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF
	   
	   # Create setup script for users
	   cat > "$PROJECT_ROOT/setup_environment.sh" << 'EOF'
#!/bin/bash
# Environment setup script for MPI Research Project

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

echo "MPI Research Project environment loaded"
echo "Project Root: \$PROJECT_ROOT"
echo "Install Directory: \$INSTALL_DIR"
EOF
	   
	   chmod +x "$PROJECT_ROOT/setup_environment.sh"
	   
	   log_success "Environment setup completed"
	   log_info "Run 'source $PROJECT_ROOT/setup_environment.sh' to load environment"
}

# Create directory structure
create_directories() {
	   log_info "Creating project directory structure..."
	   
	   mkdir -p \
	       "$BUILD_DIR" \
	       "$INSTALL_DIR" \
	       "$PROJECT_ROOT/results" \
	       "$PROJECT_ROOT/data" \
	       "$PROJECT_ROOT/docs" \
	       "$PROJECT_ROOT/tests"
	   
	   log_success "Directory structure created"
}

# Display usage information
usage() {
	   echo "Usage: $0 [OPTIONS]"
	   echo "Install dependencies for MPI Research Project"
	   echo ""
	   echo "OPTIONS:"
	   echo "  -h, --help              Show this help message"
	   echo "  --skip-system           Skip system dependency installation"
	   echo "  --skip-python           Skip Python dependency installation"
	   echo "  --skip-ns3              Skip NS-3 installation"
	   echo "  --minimal               Install only essential dependencies"
	   echo "  --full                  Install all dependencies (default)"
	   echo ""
	   echo "EXAMPLES:"
	   echo "  $0 --minimal            Install only essential dependencies"
	   echo "  $0 --skip-ns3           Install everything except NS-3"
	   echo "  $0 --full               Install all dependencies"
}

# Main installation function
main() {
	   # Parse command line arguments
	   local SKIP_SYSTEM=false
	   local SKIP_PYTHON=false
	   local SKIP_NS3=false
	   local INSTALL_TYPE="full"
	   
	   while [[ $# -gt 0 ]]; do
	       case $1 in
	           -h|--help)
	               usage
	               exit 0
	               ;;
	           --skip-system)
	               SKIP_SYSTEM=true
	               shift
	               ;;
	           --skip-python)
	               SKIP_PYTHON=true
	               shift
	               ;;
	           --skip-ns3)
	               SKIP_NS3=true
	               shift
	               ;;
	           --minimal)
	               INSTALL_TYPE="minimal"
	               shift
	               ;;
	           --full)
	               INSTALL_TYPE="full"
	               shift
	               ;;
	           *)
	               log_error "Unknown option: $1"
	               usage
	               exit 1
	               ;;
	       esac
	   done
	   
	   # Initialize log file
	   echo "MPI Research Project Installation Log" > "$LOG_FILE"
	   echo "Started: $(date)" >> "$LOG_FILE"
	   echo "======================================" >> "$LOG_FILE"
	   
	   log_info "Starting MPI Research Project dependency installation"
	   log_info "Installation type: $INSTALL_TYPE"
	   log_info "Project root: $PROJECT_ROOT"
	   
	   # Create directories
	   create_directories
	   
	   # Install system dependencies
	   if [ "$SKIP_SYSTEM" = false ] && [ "$INSTALL_TYPE" = "full" ]; then
	       install_system_dependencies
	   else
	       log_info "Skipping system dependency installation"
	   fi
	   
	   # Install Python dependencies
	   if [ "$SKIP_PYTHON" = false ]; then
	       install_python_dependencies
	   else
	       log_info "Skipping Python dependency installation"
	   fi
	   
	   # Install NS-3
	   if [ "$SKIP_NS3" = false ] && [ "$INSTALL_TYPE" = "full" ]; then
	       install_ns3_dependencies
	       install_ns3
	   else
	       log_info "Skipping NS-3 installation"
	   fi
	   
	   # Setup environment
	   setup_environment
	   
	   # Verify installations
	   if verify_installations; then
	       log_success "All dependencies installed successfully!"
	       log_info "Installation log: $LOG_FILE"
	       log_info "Next steps:"
	       log_info "1. Source environment: source $PROJECT_ROOT/setup_environment.sh"
	       log_info "2. Build project: make all"
	       log_info "3. Run tests: make test"
	   else
	       log_error "Some installations failed. Check $LOG_FILE for details."
	       exit 1
	   fi
}

# Run main function
main "$@"