# syntax=docker/dockerfile:1

# ---- build stage ----
FROM rust:1.95-slim-bookworm AS builder

WORKDIR /app

# Compile the dependency graph on a layer of its own, keyed only on the
# manifests. Editing src/ then costs a recompile of this crate alone instead of
# axum, tokio and everything under them.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
 && echo 'fn main() {}' > src/main.rs \
 && cargo build --release --locked \
 && rm -rf src

COPY src ./src
# COPY carries the host's mtimes over, which can leave the real sources looking
# older than the artifacts the dummy build just produced; without the touch,
# cargo decides there is nothing to do and the placeholder binary ships.
RUN touch src/main.rs \
 && cargo build --release --locked

# ---- runtime stage ----
# Same Debian release as the builder, so the binary meets the glibc it was
# linked against.
FROM debian:bookworm-slim AS runtime

# The server opens no files and binds an unprivileged port, so it has no reason
# to run as root.
RUN useradd --system --user-group --no-create-home app
USER app

COPY --from=builder \
     /app/target/release/learn_model_with_linear_regression_api \
     /usr/local/bin/linear-regression-api

# The binary defaults to loopback, which inside a container shuts out even its
# own host, so the image has to listen on all interfaces. Privacy is enforced by
# publishing to 127.0.0.1 on the host side, or by not publishing at all.
ENV BIND_ADDR=0.0.0.0:3000

# Where the scaling hop goes. Compose overrides this with its service name; the
# default assumes both processes share a network namespace.
ENV DATA_MANIPULATE_URL=http://0.0.0.0:3001/manipulate-datas

EXPOSE 3000

CMD ["/usr/local/bin/linear-regression-api"]
