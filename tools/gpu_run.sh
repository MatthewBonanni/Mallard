#!/usr/bin/env bash
# Build Mallard for CUDA on an A100 node and run a case there at zero priority.
#
#   tools/gpu_run.sh HOST CASE_DIR [extra Mallard args...]
#
# CASE_DIR must contain input.toml; results are copied back into it.
# The run goes through tools/zero_priority.py on the node, which refuses
# anything but A100 GPUs and kills the job the moment anyone else wants a GPU.
# Exit code 3 means the job yielded; retry later with the same command.
set -euo pipefail

host="$1"
case_dir="$2"
shift 2

case "$host" in
    *a100*) ;;
    *) echo "Refusing: $host is not an A100 node" >&2; exit 2 ;;
esac

repo="$(cd "$(dirname "$0")/.." && pwd)"
remote_root="mallard-zero-priority"
remote_case="$remote_root/cases/$(basename "$case_dir")"

ssh -o ConnectTimeout=10 "$host" mkdir -p "$remote_root/src" "$remote_case"

rsync -az --delete --exclude build --exclude 'build-*' --exclude runs --exclude .venv \
    "$repo/" "$host:$remote_root/src/"
rsync -az "$case_dir/input.toml" "$host:$remote_case/"

# Configure and build on the node without touching the GPU
ssh "$host" bash -lc "'
set -e
cd $remote_root
mkdir -p build-cuda && cd build-cuda
if [ ! -f CMakeCache.txt ]; then
    cmake ../src -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_KOKKOS=OFF \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_SERIAL=ON \
        -DCMAKE_CXX_COMPILER=\$PWD/../src/src/external/kokkos/bin/nvcc_wrapper
fi
make -j\$(nproc) Mallard
'"

set +e
ssh "$host" bash -lc "'
cd $remote_case
python3 ~/$remote_root/src/tools/zero_priority.py --log zero_priority.log -- \
    ~/$remote_root/build-cuda/src/Mallard -i input.toml $*
'"
rc=$?
set -e

rsync -az "$host:$remote_case/" "$case_dir/"
if [ $rc -eq 3 ]; then
    echo "Job yielded to another user; partial results copied back." >&2
fi
exit $rc
