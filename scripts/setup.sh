#!/bin/bash
# Autobahn Development Environment Setup Script
# Sets up complete development environment for consciousness framework

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="Autobahn"
MIN_RUST_VERSION="1.70.0"

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison function
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check Rust installation and version
check_rust() {
    log_info "Checking Rust installation..."
    
    if ! command_exists rustc; then
        log_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    
    local rust_version=$(rustc --version | cut -d' ' -f2)
    log_info "Found Rust version: $rust_version"
    
    if ! version_ge "$rust_version" "$MIN_RUST_VERSION"; then
        log_warning "Rust version $rust_version is older than required $MIN_RUST_VERSION"
        log_info "Updating Rust..."
        rustup update
    else
        log_success "Rust version is compatible"
    fi
}

# Install Rust components
install_rust_components() {
    log_info "Installing Rust components..."
    
    # Essential components
    rustup component add clippy rustfmt
    
    # Additional targets for cross-compilation if needed
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected macOS, adding additional targets..."
        rustup target add x86_64-apple-darwin aarch64-apple-darwin
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Detected Linux, adding additional targets..."
        rustup target add x86_64-unknown-linux-gnu
    fi
    
    log_success "Rust components installed"
}

# Install Cargo tools
install_cargo_tools() {
    log_info "Installing essential Cargo tools..."
    
    local tools=(
        "cargo-audit"           # Security auditing
        "cargo-outdated"        # Check for outdated dependencies
        "cargo-tree"           # Dependency tree visualization
        "cargo-watch"          # File watching for development
        "cargo-expand"         # Macro expansion
        "cargo-edit"           # Add/remove dependencies from CLI
        "cargo-udeps"          # Find unused dependencies
        "cargo-deny"           # Dependency analysis and policy enforcement
        "cargo-criterion"      # Benchmarking
        "cargo-tarpaulin"      # Code coverage (Linux only)
    )
    
    for tool in "${tools[@]}"; do
        if ! command_exists "$tool"; then
            log_info "Installing $tool..."
            if cargo install "$tool"; then
                log_success "Installed $tool"
            else
                log_warning "Failed to install $tool, continuing..."
            fi
        else
            log_info "$tool is already installed"
        fi
    done
}

# Setup project directories
setup_directories() {
    log_info "Setting up project directories..."
    
    # Create necessary directories
    mkdir -p .cargo
    mkdir -p scripts
    mkdir -p tests/integration
    mkdir -p tests/benchmarks
    mkdir -p examples
    mkdir -p docs/api
    mkdir -p target
    
    log_success "Project directories created"
}

# Create development configuration
create_dev_config() {
    log_info "Creating development configuration..."
    
    # Create .cargo/config.toml for project-specific settings
    cat > .cargo/config.toml << 'EOF'
# Cargo configuration for Autobahn project

[build]
# Use all available CPU cores for compilation
jobs = 0

[target.x86_64-unknown-linux-gnu]
# Linux-specific settings
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-apple-darwin]
# macOS Intel-specific settings
rustflags = ["-C", "target-cpu=native"]

[target.aarch64-apple-darwin]
# macOS Apple Silicon-specific settings
rustflags = ["-C", "target-cpu=native"]

[alias]
# Useful aliases for development
b = "build"
c = "check"
t = "test"
r = "run"
br = "build --release"
tr = "test --release"
fmt = "fmt --all"
clippy-all = "clippy --all-targets --all-features"

# Consciousness framework specific aliases
demo = "run --example comprehensive_example --all-features"
consciousness = "run --example complete_fire_consciousness_demo --all-features"
rag = "run --example enhanced_bio_rag_demo --all-features"
cli = "run --bin autobahn-cli --all-features"

[env]
# Environment variables for development
RUST_BACKTRACE = "1"
RUST_LOG = "debug"
EOF
    
    log_success "Development configuration created"
}

# Setup Git hooks
setup_git_hooks() {
    if [ -d ".git" ]; then
        log_info "Setting up Git hooks..."
        
        # Pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Autobahn project

set -e

echo "Running pre-commit checks..."

# Check formatting
echo "Checking code formatting..."
cargo fmt --all -- --check

# Run clippy
echo "Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# Run tests
echo "Running tests..."
cargo test --all-features

echo "Pre-commit checks passed!"
EOF
        
        chmod +x .git/hooks/pre-commit
        log_success "Git hooks installed"
    else
        log_warning "Not a Git repository, skipping Git hooks setup"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            log_info "Installing macOS dependencies with Homebrew..."
            brew install pkg-config openssl
        else
            log_warning "Homebrew not found. Please install system dependencies manually."
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            log_info "Installing Linux dependencies with apt..."
            sudo apt-get update
            sudo apt-get install -y pkg-config libssl-dev build-essential
        elif command_exists yum; then
            log_info "Installing Linux dependencies with yum..."
            sudo yum install -y pkgconfig openssl-devel gcc
        elif command_exists pacman; then
            log_info "Installing Linux dependencies with pacman..."
            sudo pacman -S --noconfirm pkgconf openssl gcc
        else
            log_warning "Package manager not found. Please install system dependencies manually."
        fi
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check if project builds
    if cargo check --all-features; then
        log_success "Project builds successfully"
    else
        log_error "Project build failed"
        exit 1
    fi
    
    # Run a quick test
    if cargo test --lib --all-features; then
        log_success "Tests pass"
    else
        log_warning "Some tests failed"
    fi
    
    log_success "Installation verification complete"
}

# Create example configuration
create_example_config() {
    log_info "Creating example configuration..."
    
    cat > autobahn.example.toml << 'EOF'
# Example configuration file for Autobahn
# Copy this to autobahn.toml and modify as needed

[system]
# Maximum ATP capacity for processing
max_atp = 15000.0

# Operating temperature in Kelvin
operating_temperature = 310.0

# Enable quantum optimization
quantum_optimization = true

[consciousness]
# Consciousness emergence threshold
emergence_threshold = 0.7

# Fire recognition sensitivity
fire_recognition_sensitivity = 0.8

[biological]
# Biological processing layers to enable
layers = ["Context", "Reasoning", "Intuition"]

# Metabolic mode
metabolic_mode = "Balanced"

[oscillatory]
# Frequency range for oscillatory processing
frequency_range = [0.1, 100.0]

# Hierarchy levels to process
hierarchy_levels = ["Molecular", "Cellular", "Organismal", "Cognitive"]

[security]
# Adversarial detection sensitivity
adversarial_sensitivity = 0.8

# Enable immune system protection
immune_protection = true

[logging]
# Log level (trace, debug, info, warn, error)
level = "info"

# Log to file
file = "autobahn.log"
EOF
    
    log_success "Example configuration created"
}

# Main setup function
main() {
    echo "================================================"
    echo "  $PROJECT_NAME Development Environment Setup  "
    echo "================================================"
    echo ""
    
    check_rust
    install_rust_components
    install_system_deps
    install_cargo_tools
    setup_directories
    create_dev_config
    setup_git_hooks
    create_example_config
    verify_installation
    
    echo ""
    echo "================================================"
    log_success "$PROJECT_NAME development environment setup complete!"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Copy autobahn.example.toml to autobahn.toml and configure"
    echo "  2. Run 'make test' to run all tests"
    echo "  3. Run 'make run-demo' to see the system in action"
    echo "  4. Run 'make help' to see all available commands"
    echo ""
    echo "Happy consciousness framework development! 🧠🔥"
}

# Run main function
main "$@" 