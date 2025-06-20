# Autobahn Makefile
# Comprehensive build, test, and development automation

.PHONY: all build test clean check fmt clippy doc bench install setup help
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := autobahn
CARGO := cargo
RUST_VERSION := 1.70.0

# Build targets
all: check test build doc ## Run all checks, tests, build, and documentation

build: ## Build the project in release mode
	$(CARGO) build --release --all-features

build-dev: ## Build the project in development mode
	$(CARGO) build --all-features

build-examples: ## Build all examples
	$(CARGO) build --examples --all-features

# Testing targets
test: ## Run all tests
	$(CARGO) test --all-features

test-unit: ## Run unit tests only
	$(CARGO) test --lib --all-features

test-integration: ## Run integration tests only
	$(CARGO) test --test '*' --all-features

test-doc: ## Run documentation tests
	$(CARGO) test --doc --all-features

test-examples: ## Test all examples
	$(CARGO) test --examples --all-features

test-verbose: ## Run tests with verbose output
	$(CARGO) test --all-features -- --nocapture

# Code quality targets
check: ## Run cargo check
	$(CARGO) check --all-features

clippy: ## Run clippy lints
	$(CARGO) clippy --all-features --all-targets -- -D warnings

clippy-fix: ## Run clippy with automatic fixes
	$(CARGO) clippy --all-features --all-targets --fix --allow-dirty -- -D warnings

fmt: ## Format code
	$(CARGO) fmt --all

fmt-check: ## Check code formatting
	$(CARGO) fmt --all -- --check

# Documentation targets
doc: ## Generate documentation
	$(CARGO) doc --all-features --no-deps

doc-open: ## Generate and open documentation
	$(CARGO) doc --all-features --no-deps --open

doc-private: ## Generate documentation including private items
	$(CARGO) doc --all-features --no-deps --document-private-items

# Benchmarking targets
bench: ## Run benchmarks
	$(CARGO) bench --all-features

bench-metabolism: ## Run metabolism-specific benchmarks
	$(CARGO) bench --all-features -- metabolism

bench-consciousness: ## Run consciousness-specific benchmarks
	$(CARGO) bench --all-features -- consciousness

# Installation and setup targets
install: ## Install the binary
	$(CARGO) install --path . --all-features

install-dev: ## Install development dependencies
	rustup component add clippy rustfmt
	cargo install cargo-audit cargo-outdated cargo-tree cargo-watch

setup: install-dev ## Complete development environment setup
	@echo "Setting up Autobahn development environment..."
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

# Utility targets
clean: ## Clean build artifacts
	$(CARGO) clean
	rm -rf target/
	rm -rf Cargo.lock

audit: ## Run security audit
	cargo audit

outdated: ## Check for outdated dependencies
	cargo outdated

tree: ## Show dependency tree
	cargo tree --all-features

watch: ## Watch for changes and run tests
	cargo watch -x "test --all-features"

watch-check: ## Watch for changes and run check
	cargo watch -x "check --all-features"

# Release targets
release-dry: ## Dry run of release
	$(CARGO) publish --dry-run --all-features

release: ## Publish to crates.io
	$(CARGO) publish --all-features

# Example targets
run-demo: ## Run the main demo
	$(CARGO) run --all-features

run-cli: ## Run the CLI interface
	$(CARGO) run --bin autobahn-cli --all-features -- --help

run-consciousness: ## Run consciousness emergence demo
	$(CARGO) run --example complete_fire_consciousness_demo --all-features

run-rag: ## Run RAG system demo
	$(CARGO) run --example enhanced_bio_rag_demo --all-features

run-comprehensive: ## Run comprehensive implementation demo
	$(CARGO) run --example comprehensive_example --all-features

# Performance targets
profile: ## Run with profiling
	$(CARGO) run --release --all-features --example comprehensive_example

profile-bench: ## Profile benchmarks
	$(CARGO) bench --all-features -- --profile-time=5

# Linting and fixing targets
fix: ## Fix common issues automatically
	$(CARGO) fix --all-features --allow-dirty

lint-all: clippy fmt-check ## Run all linting checks

fix-all: clippy-fix fmt ## Fix all automatically fixable issues

# CI/CD targets
ci-check: ## Run CI checks locally
	make fmt-check
	make clippy
	make test
	make doc

ci-full: ## Run full CI pipeline locally
	make clean
	make ci-check
	make build
	make bench

# Development helpers
dev-server: ## Start development server with auto-reload
	cargo watch -x "run --all-features"

dev-test: ## Start test watcher
	cargo watch -x "test --all-features -- --nocapture"

# Help target
help: ## Show this help message
	@echo "Autobahn Project Makefile"
	@echo "========================="
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make setup          # Set up development environment"
	@echo "  make test           # Run all tests"
	@echo "  make run-demo       # Run the main demonstration"
	@echo "  make ci-check       # Run CI checks locally"
	@echo ""

# Version information
version: ## Show version information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Rust version: $(shell rustc --version)"
	@echo "Cargo version: $(shell cargo --version)"
	@echo "Clippy version: $(shell cargo clippy --version)"
	@echo "Rustfmt version: $(shell cargo fmt --version)" 