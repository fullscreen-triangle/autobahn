#!/bin/bash
# Build script for Autobahn Consciousness Framework

set -e

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

# Configuration
PROJECT_NAME="autobahn"
BUILD_TARGETS=("x86_64-unknown-linux-gnu" "x86_64-apple-darwin" "aarch64-apple-darwin")
BUILD_TYPE="release"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --target)
            BUILD_TARGETS=("$2")
            shift 2
            ;;
        --all-targets)
            # Use default targets
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug          Build in debug mode"
            echo "  --target TARGET  Build for specific target"
            echo "  --all-targets    Build for all supported targets (default)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting Autobahn build process..."
log_info "Build type: $BUILD_TYPE"
log_info "Targets: ${BUILD_TARGETS[*]}"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    if ! command -v rustc &> /dev/null; then
        log_error "Rustc not found. Please install Rust."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Clean previous builds
clean_build() {
    log_info "Cleaning previous builds..."
    cargo clean
    log_success "Clean completed"
}

# Install required targets
install_targets() {
    log_info "Installing build targets..."
    
    for target in "${BUILD_TARGETS[@]}"; do
        log_info "Installing target: $target"
        rustup target add "$target" || {
            log_warning "Failed to install target $target, skipping..."
            continue
        }
    done
    
    log_success "Target installation completed"
}

# Build for each target
build_targets() {
    log_info "Building for targets..."
    
    for target in "${BUILD_TARGETS[@]}"; do
        log_info "Building for target: $target"
        
        if [[ "$BUILD_TYPE" == "release" ]]; then
            cargo build --release --all-features --target "$target" || {
                log_error "Build failed for target $target"
                continue
            }
        else
            cargo build --all-features --target "$target" || {
                log_error "Build failed for target $target"
                continue
            }
        fi
        
        log_success "Build completed for $target"
    done
}

# Create distribution packages
create_packages() {
    log_info "Creating distribution packages..."
    
    mkdir -p dist
    
    for target in "${BUILD_TARGETS[@]}"; do
        if [[ "$BUILD_TYPE" == "release" ]]; then
            binary_path="target/$target/release"
        else
            binary_path="target/$target/debug"
        fi
        
        if [[ -f "$binary_path/autobahn-cli" ]]; then
            package_name="${PROJECT_NAME}-${target}"
            
            # Create package directory
            mkdir -p "dist/$package_name"
            
            # Copy binary
            cp "$binary_path/autobahn-cli" "dist/$package_name/"
            
            # Copy configuration
            cp autobahn.example.toml "dist/$package_name/autobahn.toml"
            
            # Copy documentation
            cp README.md "dist/$package_name/"
            cp LICENSE "dist/$package_name/"
            cp CHANGELOG.md "dist/$package_name/"
            
            # Create archive
            cd dist
            tar -czf "$package_name.tar.gz" "$package_name"
            cd ..
            
            log_success "Package created: dist/$package_name.tar.gz"
        elif [[ -f "$binary_path/autobahn-cli.exe" ]]; then
            # Windows binary
            package_name="${PROJECT_NAME}-${target}"
            
            mkdir -p "dist/$package_name"
            cp "$binary_path/autobahn-cli.exe" "dist/$package_name/"
            cp autobahn.example.toml "dist/$package_name/autobahn.toml"
            cp README.md "dist/$package_name/"
            cp LICENSE "dist/$package_name/"
            cp CHANGELOG.md "dist/$package_name/"
            
            cd dist
            zip -r "$package_name.zip" "$package_name"
            cd ..
            
            log_success "Package created: dist/$package_name.zip"
        else
            log_warning "Binary not found for target $target"
        fi
    done
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    cargo test --all-features || {
        log_error "Tests failed"
        return 1
    }
    
    log_success "All tests passed"
}

# Generate documentation
generate_docs() {
    log_info "Generating documentation..."
    
    cargo doc --all-features --no-deps || {
        log_error "Documentation generation failed"
        return 1
    }
    
    log_success "Documentation generated"
}

# Main build process
main() {
    log_info "=== Autobahn Build Process ==="
    
    check_prerequisites
    clean_build
    install_targets
    
    # Run tests first
    if ! run_tests; then
        log_error "Build aborted due to test failures"
        exit 1
    fi
    
    build_targets
    create_packages
    generate_docs
    
    log_success "=== Build Process Complete ==="
    log_info "Distribution packages available in: dist/"
    log_info "Documentation available in: target/doc/"
}

# Run main function
main "$@" 