#!/usr/bin/env bash
set -euo pipefail

# Quick helper: compile, execute, write VK, prove a Noir circuit.
# Usage: ./scripts/provision_and_prove.sh [circuit]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIRCUIT="${1:-image_validity}"
DIR="$ROOT/proofs/$CIRCUIT"

err(){ echo "Error: $1" >&2; exit 1; }
which nargo >/dev/null 2>&1 || err "nargo not found"
which bb >/dev/null 2>&1 || err "bb not found"
[ -d "$DIR" ] || err "circuit dir missing: $DIR"

pushd "$DIR" >/dev/null
echo "Working on: $CIRCUIT"

# compile & witness
nargo compile >/dev/null
nargo execute >/dev/null

BYTE=target/${CIRCUIT}.json
WIT=target/${CIRCUIT}.gz
VK=target/vk
PROOF=target/${CIRCUIT}.bin

[ -f "$BYTE" ] || err "bytecode missing"
[ -f "$WIT" ] || err "witness missing"

bb write_vk -b "$BYTE" -o "$VK" >/dev/null
bb prove -b "$BYTE" -w "$WIT" -o "$PROOF" >/dev/null

[ -f "$PROOF" ] || err "proof not produced"
echo "OK: proof -> $PROOF"
popd >/dev/null
#!/usr/bin/env bash
set -euo pipefail

# Simple helper to compile/execute/prove/verify a Noir circuit using nargo and bb (Barretenberg)
# Usage:
#   ./scripts/provision_and_prove.sh [circuit_name]
# Example:
#   ./scripts/provision_and_prove.sh image_validity

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIRCUIT_NAME="${1:-image_validity}"
CIRCUIT_DIR="$ROOT_DIR/proofs/$CIRCUIT_NAME"

echo "Root: $ROOT_DIR"
echo "Circuit: $CIRCUIT_NAME"
echo "Circuit dir: $CIRCUIT_DIR"

command -v nargo >/dev/null 2>&1 || { echo "nargo not found in PATH. Install Noir/nargo first." >&2; exit 2; }
command -v bb >/dev/null 2>&1 || { echo "bb (Barretenberg CLI) not found in PATH. Install bb or adjust PATH." >&2; exit 2; }

if [ ! -d "$CIRCUIT_DIR" ]; then
  echo "Circuit directory not found: $CIRCUIT_DIR" >&2
  echo "Create the circuit once under proofs/$CIRCUIT_NAME (Nargo.toml + src/main.nr)" >&2
  exit 3
fi

pushd "$CIRCUIT_DIR" >/dev/null

echo "==> nargo compile"
nargo compile

echo "==> nargo execute (generates witness)"
nargo execute

BYTECODE="target/${CIRCUIT_NAME}.json"
WITNESS="target/${CIRCUIT_NAME}.gz"
VK_OUT="target/vk"
PROOF_OUT="target/${CIRCUIT_NAME}.bin"

if [ ! -f "$BYTECODE" ]; then
  echo "Bytecode not found: $BYTECODE" >&2
  popd >/dev/null
  exit 4
fi

if [ ! -f "$WITNESS" ]; then
  echo "Witness not found: $WITNESS (nargo execute should have produced it)" >&2
  popd >/dev/null
  exit 5
fi

echo "==> bb write_vk -b $BYTECODE -o $VK_OUT"
bb write_vk -b "$BYTECODE" -o "$VK_OUT"

echo "==> bb prove -b $BYTECODE -w $WITNESS -o $PROOF_OUT"
bb prove -b "$BYTECODE" -w "$WITNESS" -o "$PROOF_OUT"

if [ -f "$PROOF_OUT" ]; then
  echo "Proof written: $PROOF_OUT"
else
  echo "Proof not produced" >&2
  popd >/dev/null
  exit 6
fi

echo "==> Attempting verification (may fail if bb verify isn't supported)"
if bb verify -h >/dev/null 2>&1; then
  bb verify -b "$BYTECODE" -p "$PROOF_OUT" || echo "bb verify returned non-zero";
else
  echo "bb verify not available or doesn't support -p; skipping verification step"
fi

popd >/dev/null

echo "Done. Generated proof: $CIRCUIT_DIR/$PROOF_OUT"
echo "If you want the proof copied to ../proofs/, move it or adjust PROOF_OUT in the script."
