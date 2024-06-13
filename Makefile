TARGET += target/x86_64-unknown-linux-gnu/release/griddy
WOODS=woods
WOODS_DEST = ${WOODS}':bin/griddy'

clippy:
	cargo clippy --workspace --tests

test:
	cargo test

.PHONY: clippy

$(TARGET): src/main.rs
	RUSTFLAGS='-C target-feature=+crt-static' \
	cargo build --release --target x86_64-unknown-linux-gnu

build: $(TARGET)

woods: build
	scp -C ${TARGET} ${WOODS_DEST}
