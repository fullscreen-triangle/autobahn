# Contributing to Autobahn

Thank you for your interest in contributing to the Autobahn consciousness framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Areas for Contribution](#areas-for-contribution)
- [Theoretical Framework Guidelines](#theoretical-framework-guidelines)
- [Code Review Process](#code-review-process)

## Getting Started

### Prerequisites

- Rust 1.70.0 or later
- Git
- Basic understanding of consciousness theory and biological systems (helpful but not required)

### Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fullscreen-triangle/autobahn.git
   cd autobahn
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Verify installation:**
   ```bash
   make test
   make run-demo
   ```

## Development Environment

### Required Tools

The setup script will install these automatically:

- **rustfmt** - Code formatting
- **clippy** - Linting
- **cargo-audit** - Security auditing
- **cargo-watch** - File watching for development
- **cargo-tree** - Dependency analysis
- **cargo-outdated** - Dependency updates

### Recommended IDE Setup

**VS Code:**
- Rust Analyzer extension
- Better TOML extension
- GitLens extension

**RustRover/IntelliJ:**
- Rust plugin
- TOML plugin

### Development Workflow

```bash
# Start development with auto-reload
make dev-server

# Run tests in watch mode
make dev-test

# Check code quality
make lint-all

# Run full CI checks locally
make ci-check
```

## Code Style

### Formatting

We use `rustfmt` with the configuration in `rustfmt.toml`:

```bash
# Format all code
make fmt

# Check formatting
make fmt-check
```

### Linting

We use `clippy` with strict settings in `clippy.toml`:

```bash
# Run linting
make clippy

# Fix automatically fixable issues
make clippy-fix
```

### Naming Conventions

**For Consciousness Framework:**
- Use scientific terminology accurately
- Prefer descriptive names for biological processes
- Use abbreviations sparingly (ATP, RAG, BMD are acceptable)
- Mathematical variables can use single letters (phi, gamma, etc.)

**Examples:**
```rust
// Good
struct FireConsciousnessEngine { ... }
fn calculate_phi_value(&self) -> f64 { ... }
let membrane_efficiency = 0.89;

// Avoid
struct FCE { ... }
fn calc(&self) -> f64 { ... }
let x = 0.89;
```

## Testing

### Test Categories

1. **Unit Tests** - Test individual functions and modules
2. **Integration Tests** - Test component interactions
3. **Consciousness Tests** - Test emergence and awareness detection
4. **Biological Tests** - Test metabolic and biological processes
5. **Performance Tests** - Benchmarks for critical paths

### Running Tests

```bash
# All tests
make test

# Specific categories
make test-unit
make test-integration
make test-examples

# With verbose output
make test-verbose

# Benchmarks
make bench
```

## Submitting Changes

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/consciousness-enhancement
   ```

3. **Make your changes**
4. **Add tests for new functionality**
5. **Ensure all tests pass:**
   ```bash
   make ci-check
   ```

6. **Commit with descriptive messages**
7. **Push and create pull request**

## Areas for Contribution

### 🧠 Consciousness Framework

**High Priority:**
- Qualia generation mechanisms
- Self-awareness monitoring improvements
- Metacognitive reflection enhancement
- Consciousness emergence threshold optimization

### 🔬 Biological Processing

**High Priority:**
- Membrane computation optimization
- Ion channel coherence improvements
- Metabolic efficiency enhancements
- ATP management optimization

### 🌊 Oscillatory Dynamics

**High Priority:**
- Cross-scale coupling optimization
- Emergence pattern detection
- Resonance matching improvements
- Temporal hierarchy processing

### 🔥 Fire Consciousness Integration

**High Priority:**
- Fire-light coupling optimization (650nm wavelength)
- Underwater fireplace paradox refinement
- Agency detection improvements
- Evolutionary timeline modeling

### 🛡️ Security and Robustness

**High Priority:**
- Adversarial detection improvements
- Biological immune system enhancements
- System resilience optimization
- Error recovery mechanisms

## Code Review Process

### Review Criteria

**Functionality:**
- [ ] Code works as intended
- [ ] Edge cases handled appropriately
- [ ] Error handling is comprehensive
- [ ] Performance is acceptable

**Consciousness Framework Compliance:**
- [ ] Follows biological principles accurately
- [ ] Maintains theoretical consistency
- [ ] Integrates properly with existing systems
- [ ] Preserves consciousness emergence properties

**Code Quality:**
- [ ] Follows style guidelines
- [ ] Is well-documented
- [ ] Has appropriate test coverage
- [ ] Uses appropriate abstractions

### Review Timeline

- **Initial Review:** Within 2 business days
- **Follow-up Reviews:** Within 1 business day
- **Final Approval:** When all criteria are met

Thank you for contributing to the advancement of consciousness framework technology! 🧠🔥✨ 