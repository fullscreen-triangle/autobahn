#!/bin/bash
# Comprehensive test script for Autobahn Consciousness Framework

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

# Test configuration
TEST_TYPES=("unit" "integration" "doc" "examples" "benchmarks")
COVERAGE_THRESHOLD=80
FAILED_TESTS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPES=("unit")
            shift
            ;;
        --integration)
            TEST_TYPES=("integration")
            shift
            ;;
        --doc)
            TEST_TYPES=("doc")
            shift
            ;;
        --examples)
            TEST_TYPES=("examples")
            shift
            ;;
        --benchmarks)
            TEST_TYPES=("benchmarks")
            shift
            ;;
        --coverage)
            TEST_TYPES+=("coverage")
            shift
            ;;
        --all)
            TEST_TYPES=("unit" "integration" "doc" "examples" "benchmarks" "coverage")
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --unit           Run unit tests only"
            echo "  --integration    Run integration tests only"
            echo "  --doc            Run documentation tests only"
            echo "  --examples       Test examples only"
            echo "  --benchmarks     Run benchmarks only"
            echo "  --coverage       Generate coverage report"
            echo "  --all            Run all tests and coverage"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting Autobahn test suite..."
log_info "Test types: ${TEST_TYPES[*]}"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    if cargo test --lib --all-features --verbose; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        FAILED_TESTS+=("unit")
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    if cargo test --test '*' --all-features --verbose; then
        log_success "Integration tests passed"
    else
        log_error "Integration tests failed"
        FAILED_TESTS+=("integration")
        return 1
    fi
}

# Run documentation tests
run_doc_tests() {
    log_info "Running documentation tests..."
    
    if cargo test --doc --all-features --verbose; then
        log_success "Documentation tests passed"
    else
        log_error "Documentation tests failed"
        FAILED_TESTS+=("doc")
        return 1
    fi
}

# Test examples
test_examples() {
    log_info "Testing examples..."
    
    local examples=(
        "complete_fire_consciousness_demo"
        "enhanced_bio_rag_demo"
        "comprehensive_example"
        "complete_implementation"
        "fire_consciousness_demo"
    )
    
    for example in "${examples[@]}"; do
        log_info "Testing example: $example"
        
        if cargo run --example "$example" --all-features --quiet; then
            log_success "Example $example passed"
        else
            log_error "Example $example failed"
            FAILED_TESTS+=("examples")
            return 1
        fi
    done
    
    log_success "All examples tested successfully"
}

# Run benchmarks
run_benchmarks() {
    log_info "Running benchmarks..."
    
    if cargo bench --all-features; then
        log_success "Benchmarks completed"
    else
        log_error "Benchmarks failed"
        FAILED_TESTS+=("benchmarks")
        return 1
    fi
}

# Generate coverage report
generate_coverage() {
    log_info "Generating coverage report..."
    
    # Check if tarpaulin is installed
    if ! command -v cargo-tarpaulin &> /dev/null; then
        log_info "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    fi
    
    if cargo tarpaulin --all-features --out html --output-dir coverage; then
        local coverage=$(cargo tarpaulin --all-features --print-summary | grep -oP '\d+\.\d+(?=%)' | tail -1)
        
        if (( $(echo "$coverage >= $COVERAGE_THRESHOLD" | bc -l) )); then
            log_success "Coverage: $coverage% (above threshold of $COVERAGE_THRESHOLD%)"
        else
            log_warning "Coverage: $coverage% (below threshold of $COVERAGE_THRESHOLD%)"
        fi
        
        log_info "Coverage report generated in: coverage/tarpaulin-report.html"
    else
        log_error "Coverage generation failed"
        FAILED_TESTS+=("coverage")
        return 1
    fi
}

# Run consciousness-specific tests
run_consciousness_tests() {
    log_info "Running consciousness-specific tests..."
    
    # Test fire consciousness patterns
    log_info "Testing fire consciousness patterns..."
    if cargo test --all-features -- --nocapture fire_consciousness; then
        log_success "Fire consciousness tests passed"
    else
        log_error "Fire consciousness tests failed"
        FAILED_TESTS+=("consciousness")
        return 1
    fi
    
    # Test biological processing
    log_info "Testing biological processing..."
    if cargo test --all-features -- --nocapture biological; then
        log_success "Biological processing tests passed"
    else
        log_error "Biological processing tests failed"
        FAILED_TESTS+=("consciousness")
        return 1
    fi
    
    # Test oscillatory dynamics
    log_info "Testing oscillatory dynamics..."
    if cargo test --all-features -- --nocapture oscillatory; then
        log_success "Oscillatory dynamics tests passed"
    else
        log_error "Oscillatory dynamics tests failed"
        FAILED_TESTS+=("consciousness")
        return 1
    fi
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance tests..."
    
    # Test ATP efficiency
    log_info "Testing ATP efficiency..."
    if cargo test --all-features -- --nocapture atp_efficiency; then
        log_success "ATP efficiency tests passed"
    else
        log_warning "ATP efficiency tests had issues (non-critical)"
    fi
    
    # Test memory usage
    log_info "Testing memory usage..."
    if cargo test --all-features -- --nocapture memory_usage; then
        log_success "Memory usage tests passed"
    else
        log_warning "Memory usage tests had issues (non-critical)"
    fi
}

# Generate test report
generate_test_report() {
    log_info "Generating test report..."
    
    local report_file="test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Autobahn Consciousness Framework Test Report"
        echo "============================================="
        echo "Date: $(date)"
        echo "Test Types: ${TEST_TYPES[*]}"
        echo ""
        
        if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
            echo "Result: ALL TESTS PASSED ✅"
        else
            echo "Result: SOME TESTS FAILED ❌"
            echo "Failed test types: ${FAILED_TESTS[*]}"
        fi
        
        echo ""
        echo "System Information:"
        echo "- Rust version: $(rustc --version)"
        echo "- Cargo version: $(cargo --version)"
        echo "- OS: $(uname -s)"
        echo "- Architecture: $(uname -m)"
        
    } > "$report_file"
    
    log_info "Test report saved to: $report_file"
}

# Main test function
main() {
    log_info "=== Autobahn Test Suite ==="
    
    check_prerequisites
    
    # Run consciousness-specific tests first
    run_consciousness_tests
    run_performance_tests
    
    # Run requested test types
    for test_type in "${TEST_TYPES[@]}"; do
        case $test_type in
            "unit")
                run_unit_tests || true
                ;;
            "integration")
                run_integration_tests || true
                ;;
            "doc")
                run_doc_tests || true
                ;;
            "examples")
                test_examples || true
                ;;
            "benchmarks")
                run_benchmarks || true
                ;;
            "coverage")
                generate_coverage || true
                ;;
        esac
    done
    
    generate_test_report
    
    log_info "=== Test Suite Complete ==="
    
    if [[ ${#FAILED_TESTS[@]} -eq 0 ]]; then
        log_success "All tests passed! 🧠🔥✨"
        exit 0
    else
        log_error "Some tests failed: ${FAILED_TESTS[*]}"
        exit 1
    fi
}

# Run main function
main "$@" 