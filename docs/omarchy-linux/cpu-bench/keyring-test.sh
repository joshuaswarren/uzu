#!/bin/bash
# Run on the Linux node, from a uzu checkout that has the keyring change. Log: $UZU_PORT/keyring-test.log
B=${UZU_PORT:-$HOME/scratch/uzu-port}
export RUSTUP_HOME=$B/rustup CARGO_HOME=$B/cargo RUSTUP_TOOLCHAIN=nightly CARGO_TARGET_DIR=$B/target
export PATH=$B/cargo/bin:$PATH
cd "${UZU_DIR:-$B/uzu}" || exit 2
{
  echo "== start $(date -u +%T) session bus: ${DBUS_SESSION_BUS_ADDRESS:-none}"
  nice -n 10 cargo test -p uzu --no-default-features --features backend-cpu --test settings_linux -- --nocapture 2>&1 | tail -25
  echo "== end rc=${PIPESTATUS[0]} $(date -u +%T)"
} > "$B/keyring-test.log" 2>&1
