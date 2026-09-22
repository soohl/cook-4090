"""Offline runtime controls; installation/download commands remain separate."""

import ctypes
import errno
import ipaddress
import os
import socket
import sys


def environment():
    return {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "GRADIO_ANALYTICS_ENABLED": "False",
        "GRADIO_RUN_HISTORY": "false",
        "DO_NOT_TRACK": "1", "WANDB_MODE": "disabled", "NO_PROXY": "*", "no_proxy": "*",
    }


def is_local(host):
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def restrict_ui_network():
    """Python socket audit hooks cover Gradio/httpx/urllib; allow only local engines."""
    os.environ.update(environment())
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)

    def audit(event, args):
        if event in {"socket.connect", "socket.sendto"}:
            sock, address = args
            if sock.family in {socket.AF_INET, socket.AF_INET6} and not is_local(address[0]):
                raise PermissionError("Offline UI: outgoing connections must use loopback")
        if event == "socket.getaddrinfo":
            host = args[0]
            if host not in (None, "", "0.0.0.0", "::") and not is_local(host):
                raise PermissionError("Offline UI: external name resolution is disabled")

    sys.addaudithook(audit)


def restrict_worker_network(local_ipc=False):
    """Block connects and Internet datagrams; local IPC uses the native connect guard."""
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]

    class Compare(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int), ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]

    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Compare)]
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("Cannot initialize offline worker filter")
    deny = 0x00050000 | errno.EPERM
    try:
        for name in (() if local_ipc else (b"connect",)):
            if lib.seccomp_rule_add(ctx, deny, lib.seccomp_syscall_resolve_name(name), 0) < 0:
                raise RuntimeError("Cannot deny outbound connections")
        for family in (socket.AF_INET, socket.AF_INET6):
            for kind in (socket.SOCK_DGRAM, socket.SOCK_RAW):
                # Ignore SOCK_CLOEXEC / SOCK_NONBLOCK flags when comparing socket type.
                rules = (Compare * 2)(Compare(0, 4, family, 0), Compare(1, 7, 15, kind))
                if lib.seccomp_rule_add_array(ctx, deny, lib.seccomp_syscall_resolve_name(b"socket"), 2, rules) < 0:
                    raise RuntimeError("Cannot deny outbound datagrams")
        if lib.seccomp_load(ctx) < 0:
            raise RuntimeError("Cannot load offline worker filter")
    finally:
        lib.seccomp_release(ctx)


if __name__ == "__main__":
    # If the UI crashes, terminate its direct inference child as well.
    import signal
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() == 1:
        raise RuntimeError("Cannot attach worker lifetime to UI")
    os.environ.update(environment())
    command = sys.argv[1:]
    local_ipc = command[0] == "--local-ipc"
    if local_ipc:
        from pathlib import Path
        command = command[1:]
        guard = Path(os.environ["SGLANG_OFFLINE_LIB"]).resolve(strict=True)
        ctypes.CDLL(str(guard)).connect  # Fail before exec if the guard is not loadable.
        os.environ["LD_PRELOAD"] = str(guard) + (":" + os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else "")
        os.environ.update(NCCL_SOCKET_IFNAME="lo", GLOO_SOCKET_IFNAME="lo")
    restrict_worker_network(local_ipc=local_ipc)
    os.execvpe(command[0], command, os.environ)
