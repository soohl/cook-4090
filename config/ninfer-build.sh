#!/usr/bin/env bash
set -euo pipefail

cmake --build /build -j --target ninfer ninfer-serve

# Bundle runtime dependencies. Keep glibc and the GPU driver supplied by the host.
mkdir -p /build/lib
for binary in /build/apps/ninfer /build/apps/ninfer-serve; do
    dependencies=$(ldd "$binary")
    if [[ "$dependencies" == *"not found"* ]]; then
        printf '%s\n' "$dependencies" >&2
        exit 1
    fi
    while read -r library; do
        case "$(basename "$library")" in
            libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libcuda.so.*|libnvidia-*) continue ;;
        esac
        destination="/build/lib/$(basename "$library")"
        [[ "$library" -ef "$destination" ]] || cp -L "$library" "$destination"
    done < <(awk '/=> \// {print $3}' <<< "$dependencies")
    patchelf --set-rpath '$ORIGIN/../lib' "$binary"
done
for library in /build/lib/*.so*; do
    patchelf --set-rpath '$ORIGIN' "$library"
done
