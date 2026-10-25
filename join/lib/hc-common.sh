# shellcheck shell=bash
# Hypercompute one-click bootstrap, sourced by the .command and .sh wrappers in join/.
# bash 3.2 compatible (macOS): no mapfile, no ${x,,}, no associative arrays.
# Steps: find the checkout, find uv, create the Python environment once (it asks first), then hand over to
# join/lib/oneclick.py. With --dry-run it prints each command instead of running it.
# It never downloads uv and never pipes an installer. It never echoes an invite, and it stops before uv when
# an invite is passed on the command line.

HC_UV_DOCS="https://docs.astral.sh/uv/getting-started/installation/"

hc_mask() {
  # Hide anything shaped like an invite ticket or code before it is shown.
  printf '%s\n' "$*" | sed -E 's/[A-Za-z0-9_-]{30,}/<r>/g; s/[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}/<code>/g'
}

hc_is_dry_run() {
  local hc_arg
  for hc_arg in "$@"; do
    if [ "$hc_arg" = "--dry-run" ]; then return 0; fi
  done
  return 1
}

hc_has_invite_arg() {
  # True when an argument looks like part of an invite: the flags, a bare ticket or a bare code.
  local hc_arg
  for hc_arg in "$@"; do
    case "$hc_arg" in
      --ticket | --ticket=* | --code | --code=*) return 0 ;;
    esac
    if printf '%s\n' "$hc_arg" | grep -Eq '^[A-Za-z0-9_-]{30,}$|^[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}$'; then
      return 0
    fi
  done
  return 1
}

hc_find_repo() {
  local hc_dir="${HC_REPO:-$HC_HERE/..}"
  HC_REPO_DIR="$(cd "$hc_dir" 2>/dev/null && pwd)" || HC_REPO_DIR=""
  if [ -z "$HC_REPO_DIR" ] || [ ! -f "$HC_REPO_DIR/backend/hypercompute/cli.py" ] ||
    ! grep -Eq '^name = "hypercompute"[[:space:]]*$' "$HC_REPO_DIR/pyproject.toml" 2>/dev/null; then
    echo "This folder is not inside a Hypercompute checkout"
    return 1
  fi
}

hc_find_uv() {
  HC_UV="$(command -v uv 2>/dev/null)" || HC_UV=""
  if [ -z "$HC_UV" ] && [ -x "$HOME/.local/bin/uv" ]; then HC_UV="$HOME/.local/bin/uv"; fi
  if [ -z "$HC_UV" ]; then
    echo "uv is not installed. Install it once from $HC_UV_DOCS and then open this again."
    return 1
  fi
}

hc_ensure_venv() {
  # $1 is 1 for a dry run. Returns 0 when ready, 1 to stop, 3 when a dry run stops here.
  local hc_answer
  if [ -d "$HC_REPO_DIR/.venv" ]; then return 0; fi
  echo "The first run downloads Python packages into $HC_REPO_DIR/.venv. This takes a few minutes."
  echo "If this computer has no Python 3.12, uv also downloads Python 3.12 into its own folder outside"
  echo "Hypercompute. That copy stays after you delete the Hypercompute folder."
  if [ "$1" = 1 ]; then
    echo "[dry-run] would ask: Continue? [y/N]"
    echo "[dry-run] would run: cd \"$HC_REPO_DIR\" && \"$HC_UV\" sync --frozen"
    return 3
  fi
  printf 'Continue? [y/N] '
  read -r hc_answer || hc_answer=""
  case "$hc_answer" in
    y | Y | yes | Yes | YES) ;;
    *)
      echo "Stopped. Nothing was installed."
      return 1
      ;;
  esac
  (cd "$HC_REPO_DIR" && "$HC_UV" sync --frozen)
}

hc_run() {
  local hc_action="$1" hc_dry=0 hc_rc
  shift
  if hc_has_invite_arg "$@"; then
    echo "Run this with nothing after it, or only --dry-run. It asks you to paste the invite."
    return 2
  fi
  if hc_is_dry_run "$@"; then hc_dry=1; fi
  hc_find_repo || return 1
  hc_find_uv || return 1
  hc_ensure_venv "$hc_dry"
  hc_rc=$?
  if [ "$hc_rc" = 3 ]; then
    echo "[dry-run] stopping here: the next steps need the Python environment."
    return 0
  fi
  if [ "$hc_rc" != 0 ]; then return "$hc_rc"; fi
  if [ "$hc_dry" = 1 ]; then
    hc_mask "[dry-run] next: cd \"$HC_REPO_DIR\" && \"$HC_UV\" run --no-sync python join/lib/oneclick.py $hc_action --repo \"$HC_REPO_DIR\" $*"
    echo "[dry-run] that helper runs in dry-run mode too: it prints its commands and launches nothing."
  fi
  cd "$HC_REPO_DIR" || return 1
  "$HC_UV" run --no-sync python join/lib/oneclick.py "$hc_action" --repo "$HC_REPO_DIR" "$@"
}

hc_pause() {
  # .command always pauses so the Terminal window stays readable; .sh only when run from a terminal.
  local hc_ignored
  if [ "${HC_NO_PAUSE:-}" = 1 ]; then return 0; fi
  case "$0" in
    *.command) ;;
    *) if [ ! -t 0 ]; then return 0; fi ;;
  esac
  printf 'Press Return to close this window'
  read -r hc_ignored || true
  echo
}

hc_main() {
  local hc_status
  hc_run "$@"
  hc_status=$?
  hc_pause
  return "$hc_status"
}
