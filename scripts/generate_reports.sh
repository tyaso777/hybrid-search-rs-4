#!/usr/bin/env bash
set -euo pipefail

# Generate vulnerability and license reports for the workspace (Bash)
#
# Usage (from repo root):
#   bash scripts/generate_reports.sh
#
# Options:
#   --install-tools     Install cargo-audit and cargo-license if missing
#   --out-dir <path>    Output directory (default: reports)
#   --include-dev-deps  Include dev-dependencies in license report (default: false)
#   --no-all-features   Do not pass --all-features to cargo license

INSTALL_TOOLS=0
OUT_DIR="reports"
INCLUDE_DEV_DEPS=0
NO_ALL_FEATURES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-tools) INSTALL_TOOLS=1; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --include-dev-deps) INCLUDE_DEV_DEPS=1; shift ;;
    --no-all-features) NO_ALL_FEATURES=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

ensure_tool() {
  local bin="$1"; local crate="$2"
  if ! command -v "$bin" >/dev/null 2>&1; then
    if [[ "$INSTALL_TOOLS" == "1" ]]; then
      echo "Installing $crate..."
      cargo install "$crate" --locked
    else
      echo "Missing $bin. Install it or rerun with --install-tools." >&2
      exit 1
    fi
  fi
}

echo "=== Generating reports (bash) ==="

ensure_tool cargo-audit cargo-audit
ensure_tool cargo-license cargo-license
ensure_tool cargo-about cargo-about

mkdir -p "$OUT_DIR"

# 1) Vulnerability report (cargo-audit)
echo "-> Running cargo audit (text + JSON)"
cargo audit --json >"$OUT_DIR/cargo-audit.json"
cargo audit >"$OUT_DIR/cargo-audit.txt"

# 2) License report (aggregate workspace members)
echo "-> Collecting license info across workspace"

license_args=()
[[ "$NO_ALL_FEATURES" == "0" ]] && license_args+=(--all-features)
[[ "$INCLUDE_DEV_DEPS" == "0" ]] && license_args+=(--avoid-dev-deps)

if command -v jq >/dev/null 2>&1; then
  # jq-based aggregation
  manifests=$(cargo metadata -q --format-version 1 | \
    jq -r '.workspace_members as $w | [.packages[] | select(.id as $id | $w | index($id)) | .manifest_path] | .[]')

  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT

  idx=0
  while IFS= read -r manifest; do
    echo "   - cargo license for ${manifest}"
    cargo license --manifest-path "$manifest" "${license_args[@]}" --json >"$tmpdir/$idx.json"
    idx=$((idx+1))
  done <<< "$manifests"

  jq -s 'add | unique_by(.name + "@" + .version)' "$tmpdir"/*.json >"$OUT_DIR/license.json"

  # Text summary
  echo "License Summary (unique crates across workspace)" >"$OUT_DIR/license.txt"
  jq -r '[.[] | .license] | group_by(.) | sort_by(length) | reverse | .[] | "\(. | length)\t\(.[0])"' \
    "$OUT_DIR/license.json" >>"$OUT_DIR/license.txt"
  printf "\nCrates:\n" >>"$OUT_DIR/license.txt"
  jq -r '. | sort_by(.name, .version) | .[] | "\(.name) \(.version)\t\(.license)"' \
    "$OUT_DIR/license.json" >>"$OUT_DIR/license.txt"
else
  # Fallback without jq: use cargo metadata --no-deps and TSV output
  manifests=$(cargo metadata -q --no-deps --format-version 1 | grep -o '"manifest_path":"[^"]*"' | sed 's/.*:"//; s/"$//')

  tmpfile=$(mktemp)
  trap 'rm -f "$tmpfile"' EXIT

  while IFS= read -r manifest; do
    echo "   - cargo license for ${manifest}"
    cargo license --manifest-path "$manifest" "${license_args[@]}" --tsv >>"$tmpfile"
  done <<< "$manifests"

  # Deduplicate by name+version
  # TSV columns: name<TAB>version<TAB>license<...>
  sort -u -t$'\t' -k1,1 -k2,2 "$tmpfile" >"$OUT_DIR/license.tsv"

  # Write text summary
  {
    echo "License Summary (unique crates across workspace)"
    cut -f3 "$OUT_DIR/license.tsv" | sort | uniq -c | sort -nr | awk '{print $1"\t"substr($0, index($0,$2))}'
    echo
    echo "Crates:"
    awk -F'\t' '{print $1" "$2"\t"$3}' "$OUT_DIR/license.tsv" | sort
  } >"$OUT_DIR/license.txt"

  # Generate a simple JSON array from TSV (name, version, license)
  awk -F'\t' 'BEGIN{print "["} {printf "%s{\"name\":\"%s\",\"version\":\"%s\",\"license\":\"%s\"}", NR==1?"":"," , $1, $2, $3} END{print "]"}' \
    "$OUT_DIR/license.tsv" >"$OUT_DIR/license.json"
fi

## 3) THIRD-PARTY-NOTICES (exclude workspace crates when jq is available)
echo "-> Generating THIRD-PARTY-NOTICES.txt"
tp_file="$OUT_DIR/THIRD-PARTY-NOTICES.txt"
{
  echo "Third-Party Notices"
  echo
  date -u '+Generated: %Y-%m-%d %H:%M:%S UTC' || true
  echo "This file lists third-party Rust crates used by this project, with their licenses and repositories."
  echo "This is an informational summary; refer to each crate for authoritative license text."
  echo
} >"$tp_file"

if command -v jq >/dev/null 2>&1; then
  ws_names=$(cargo metadata -q --format-version 1 | jq -r '.workspace_members as $w | [.packages[] | select(.id as $id | $w | index($id)) | .name]')
  {
    echo "Summary by license (third-party only)"
    jq --argjson ws "$ws_names" -r '[.[] | select((.name | IN($ws[])) | not) | .license] | group_by(.) | sort_by(length) | reverse | .[] | "\(. | length)\t\(.[0])"' \
      "$OUT_DIR/license.json"
    echo
    echo "Third-party crates:"
    jq --argjson ws "$ws_names" -r '. | sort_by(.name, .version) | .[] | select((.name | IN($ws[])) | not) | "- \(.name) \(.version)  [\(.license // "UNKNOWN")]  \(.repository // "")"' \
      "$OUT_DIR/license.json"
  } >>"$tp_file"
else
  {
    echo "Summary by license (unfiltered)"
    cut -f3 "$OUT_DIR/license.tsv" | sort | uniq -c | sort -nr | awk '{print $1"\t"substr($0, index($0,$2))}'
    echo
    echo "Crates (unfiltered):"
    awk -F '\t' '{print "- "$1" "$2"  ["$3"]  "}' "$OUT_DIR/license.tsv" | sort
  } >>"$tp_file"
fi

echo "=== Done ==="
echo "- Vulnerabilities: $OUT_DIR/cargo-audit.txt"
echo "- Vulnerabilities (JSON): $OUT_DIR/cargo-audit.json"
echo "- Licenses: $OUT_DIR/license.txt"
echo "- Licenses (JSON): $OUT_DIR/license.json"
echo "- Third-party notices: $tp_file"

## 4) cargo-about: consolidated licenses including license texts
echo "-> Generating cargo-about (JSON + notices)"
about_json="$OUT_DIR/about.json"
cargo about generate --format json --all-features --workspace -o "$about_json"

full_tp_file="$OUT_DIR/THIRD-PARTY-LICENSES.txt"
if command -v jq >/dev/null 2>&1; then
  {
    echo "THIRD-PARTY LICENSES (cargo-about)"
    echo
    date -u '+Generated: %Y-%m-%d %H:%M:%S UTC' || true
    echo "This file contains license texts for third-party crates used by this project."
    echo "Workspace/local crates are excluded."
    echo

    # For each overview license, list crates with matching license and then print license text
    jq -r '
      . as $root |
      .overview | .[] | {id, name, text} as $lic |
      (
        "=== \($lic.name) (\($lic.id)) ===\n" +
        "Used by:\n" +
        ($root.crates
          | map(select(.package.source != null) | select(.license == $lic.id))
          | sort_by(.package.name, .package.version)
          | map("- " + .package.name + " " + .package.version + "  " + (.package.repository // ""))
          | join("\n")
        ) +
        "\n\nLicense text:\n" +
        ("-" * 79) + "\n" +
        ($lic.text // "N/A") + "\n" +
        ("-" * 79) + "\n\n"
      )
    ' "$about_json"
  } >"$full_tp_file"
else
  # Minimal fallback: just dump JSON path to inform user
  echo "jq not found; skipping consolidated license texts (cargo-about)." >>"$full_tp_file"
  echo "See $about_json for raw data." >>"$full_tp_file"
fi
echo "- Cargo-about full licenses: $full_tp_file"
