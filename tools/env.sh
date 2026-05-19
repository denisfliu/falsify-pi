# Falsify environment shims.
#
# `source tools/env.sh` (or `. tools/env.sh`) from the repo root before any
# command that may JIT-build a torch CUDA extension — `gsplat_cuda`, the
# nerfstudio CUDA ops, FiGS CUDA modules, etc.
#
# Why this exists
# ---------------
# CUDA 12.0's nvcc (V12.0.140) officially supports gcc up to 12. Ubuntu
# 24.04 ships gcc 13.3.0 as `/usr/bin/gcc`; the combo of nvcc 12.0 + gcc 13
# rejects the pybind11 template syntax in torch's headers with:
#
#   pybind11/cast.h:45: error: expected template-name before '<' token
#
# Pinning nvcc's host compiler to gcc-11 (also installed on the system) fixes
# the JIT rebuild. The previously-cached `.so` keeps working until the cache
# is invalidated (torch upgrade, gsplat reinstall, `rm -rf
# ~/.cache/torch_extensions/...`, etc.), which is why the failure keeps
# resurfacing without these exports.
#
# What this exports
# -----------------
# CC / CXX           — host compilers used by torch.utils.cpp_extension
# NVCC_PREPEND_FLAGS — forwarded into every nvcc invocation; the `-ccbin`
#                      flag tells nvcc which host compiler to drive.
#
# Safe to source repeatedly; safe to source even when the cache is already
# warm (the exports only matter at build time).
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

# acados (FiGS' OCP solver) — falsify's submodule isn't built; SousVide's is.
# `acados_template` Python finds the prebuilt .so / t_renderer via these env
# vars. Override $ACADOS_SOURCE_DIR if you build falsify's own copy.
: "${ACADOS_SOURCE_DIR:=/home/dfliu/code/SousVide/external/FiGS/acados}"
export ACADOS_SOURCE_DIR
case ":${LD_LIBRARY_PATH:-}:" in
  *":$ACADOS_SOURCE_DIR/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

# Also keep the PYTHONPATH workaround for the symlinked SousVide venv that
# doesn't register falsify as editable. Idempotent on repeated sourcing.
_FALSIFY_REPO="${BASH_SOURCE[0]:-${(%):-%x}}"
_FALSIFY_REPO="$(cd "$(dirname "$_FALSIFY_REPO")/.." && pwd)"
case ":${PYTHONPATH:-}:" in
  *":$_FALSIFY_REPO/src:"*) ;;
  *) export PYTHONPATH="$_FALSIFY_REPO/src:$_FALSIFY_REPO/external/FiGS/src:$_FALSIFY_REPO/external/splatnav${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
unset _FALSIFY_REPO
