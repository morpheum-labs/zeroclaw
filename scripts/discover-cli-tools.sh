#!/usr/bin/env bash
# Discover the same CLI tools as src/tools/cli_discovery.rs (KNOWN_CLIS).
# Step 1: list available tools. Step 2: optional symlinks — [c] KNOWN_CLIS only,
# [r] all executables under $HOME/.cargo/bin/ (verbose).
# Portable: Linux + macOS; uses the current shell environment (PATH, etc.).
set -euo pipefail

os="$(uname -s)"
echo "OS: ${os}"
echo "PATH (first 200 chars): ${PATH:0:200}$([ "${#PATH}" -gt 200 ] && echo '…' || true)"
echo ""

# Parallel arrays (bash 3.2–compatible): discovered tool names and absolute paths.
tool_names=()
tool_paths=()

probe_and_collect() {
  local name="$1"
  shift
  local -a args=("$@")
  local path
  if ! path="$(command -v "$name" 2>/dev/null)"; then
    echo "[missing] $name (not on PATH)"
    return 1
  fi
  local out
  if ! out="$("$name" "${args[@]}" 2>&1)"; then
    echo "[broken] $name at $path (version probe failed)"
    return 1
  fi
  local first
  first="$(printf '%s\n' "$out" | head -n 1)"
  echo "[ok] $name ($path) — $first"
  tool_names+=("$name")
  tool_paths+=("$path")
  return 0
}

echo "=== Step 1: Discover CLI tools (max 16 built-in) ==="
echo ""

# Order matches src/tools/cli_discovery.rs KNOWN_CLIS
probe_and_collect git --version || true
probe_and_collect python --version || true
probe_and_collect python3 --version || true
probe_and_collect node --version || true
probe_and_collect npm --version || true
probe_and_collect pip --version || true
probe_and_collect pip3 --version || true
probe_and_collect docker --version || true
probe_and_collect cargo --version || true
probe_and_collect make --version || true
probe_and_collect kubectl version --client --short || true
probe_and_collect rustc --version || true
probe_and_collect claude --version || true
probe_and_collect gemini --version || true
probe_and_collect kilo --version || true
probe_and_collect gws --version || true

found="${#tool_names[@]}"
echo ""
echo "--- Available as tools (${found}) ---"
if [[ "$found" -eq 0 ]]; then
  echo "(none)"
else
  i=1
  while [[ "$i" -le "$found" ]]; do
    idx=$((i - 1))
    echo "  ${i}. ${tool_names[$idx]} -> ${tool_paths[$idx]}"
    i=$((i + 1))
  done
fi
echo ""
echo "Summary: ${found} discovered (max 16 built-in)."
echo ""

echo "=== Step 2: Symlinks (optional) ==="
echo "Create symlinks with ln -sf."
echo "  [c] continue — link each *discovered* tool into a target directory"
echo "  [r] cargo-bin — link *all* executables from \$HOME/.cargo/bin/ (verbose)"
echo "  [q] exit     — quit now"
read -r -p "Your choice [c/r/q] (default q): " choice
choice="${choice:-q}"
case "$(printf '%s' "$choice" | tr '[:upper:]' '[:lower:]')" in
c|continue|y|yes)
  read -r -p "Target directory (symlinks: <dir>/<toolname>): " target_dir
  if [[ -z "$target_dir" ]]; then
    echo "No directory given; exiting."
    exit 1
  fi
  # Trim trailing slash for consistent display
  target_dir="${target_dir%/}"
  mkdir -p "$target_dir"
  created=0
  idx=0
  while [[ "$idx" -lt "$found" ]]; do
    src="${tool_paths[$idx]}"
    name="${tool_names[$idx]}"
    dest="${target_dir}/${name}"
    if ln -sf "$src" "$dest"; then
      echo "  ln -sf \"$src\" -> \"$dest\""
      created=$((created + 1))
    else
      echo "  [fail] could not link $name -> $dest" >&2
    fi
    idx=$((idx + 1))
  done
  echo ""
  echo "Created ${created} symlink(s) under ${target_dir}."
  ;;
r|rust|cargo-bin)
  cargo_bin="${HOME}/.cargo/bin"
  echo ""
  echo "[verbose] Option r: bulk symlink from ${cargo_bin}"
  if [[ ! -d "$cargo_bin" ]]; then
    echo "[verbose] ERROR: directory does not exist: ${cargo_bin}" >&2
    exit 1
  fi
  read -r -p "Target directory (symlinks: <dir>/<basename>): " target_dir
  if [[ -z "$target_dir" ]]; then
    echo "No directory given; exiting."
    exit 1
  fi
  target_dir="${target_dir%/}"
  echo "[verbose] mkdir -p \"${target_dir}\""
  mkdir -p "$target_dir"
  created=0
  skipped=0
  _ng_restore=
  if shopt -q nullglob 2>/dev/null; then
    _ng_restore=on
  else
    _ng_restore=off
  fi
  shopt -s nullglob 2>/dev/null || true
  for src in "${cargo_bin}"/*; do
    [[ -e "$src" ]] || continue
    if [[ -d "$src" ]]; then
      echo "[verbose] skip (directory): $src"
      skipped=$((skipped + 1))
      continue
    fi
    # Regular files and symlinks to binaries/scripts in .cargo/bin
    if [[ ! -f "$src" ]]; then
      echo "[verbose] skip (not a file): $src"
      skipped=$((skipped + 1))
      continue
    fi
    if [[ ! -x "$src" ]]; then
      echo "[verbose] skip (not executable): $src"
      skipped=$((skipped + 1))
      continue
    fi
    name="$(basename "$src")"
    dest="${target_dir}/${name}"
    echo "[verbose] ln -sf \"$src\" \"$dest\""
    if ln -sf "$src" "$dest"; then
      echo "[verbose] OK -> ${name}"
      created=$((created + 1))
    else
      echo "[verbose] FAIL -> ${name}" >&2
    fi
  done
  if [[ "$_ng_restore" == "off" ]]; then
    shopt -u nullglob 2>/dev/null || true
  fi
  echo ""
  if [[ "$created" -eq 0 ]] && [[ "$skipped" -eq 0 ]]; then
    echo "[verbose] No files matched in ${cargo_bin} (empty or glob not expanded)."
  fi
  echo "[verbose] Done: ${created} symlink(s) created, ${skipped} skipped, under ${target_dir}."
  ;;
*)
  echo "Exiting without creating symlinks."
  exit 0
  ;;
esac
