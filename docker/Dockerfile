# Multi-stage Docker build for Autobahn Consciousness Framework

# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY examples ./examples
COPY benches ./benches

# Copy configuration files
COPY clippy.toml rustfmt.toml ./

# Build the application
RUN cargo build --release --all-features

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false autobahn

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/autobahn-cli /usr/local/bin/autobahn-cli

# Copy configuration
COPY autobahn.example.toml /app/autobahn.toml

# Create data directory
RUN mkdir -p /app/data && chown autobahn:autobahn /app/data

# Switch to non-root user
USER autobahn

# Expose port (if needed for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD autobahn-cli --version || exit 1

# Default command
CMD ["autobahn-cli", "--help"] 