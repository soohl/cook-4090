"""Docker lifecycle and private endpoint setup. Never evaluate .env as shell code."""

import base64
import ipaddress
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys

from . import ROOT


def refresh_docker_group(module):
    """Use existing account membership when this shell predates Docker setup."""
    if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
        return
    socket = Path("/var/run/docker.sock")
    if not socket.exists() or os.access(socket, os.R_OK | os.W_OK):
        return
    import grp
    import pwd
    group = grp.getgrgid(socket.stat().st_gid)
    account = pwd.getpwuid(os.getuid())
    if group.gr_name != "docker" or group.gr_gid in {*os.getgroups(), os.getgid()}:
        return
    if group.gr_gid not in os.getgrouplist(account.pw_name, account.pw_gid) or not shutil.which("sg"):
        return
    print("  INFO   Docker access             Refreshing existing docker group membership", flush=True)
    command = shlex.join([sys.executable, "-m", module, *sys.argv[1:]])
    os.execvp("sg", ["sg", "docker", "-c", command])


def load_env(path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("Use KEY=value lines in .env")
        if value.startswith(("'", '"')) or any(c.isspace() for c in value) or "$" in value or "`" in value:
            raise ValueError(f"Use an unquoted literal value without whitespace or interpolation for {key}")
        values[key] = value
    return values


def validate_bind(value):
    address = ipaddress.ip_address(value)
    allowed = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]
    if address.version != 4 or not any(address in network for network in allowed):
        raise ValueError("CODING_BIND must be loopback or an assigned RFC1918 IPv4 address")
    return str(address)


def initialize(path):
    values = {"FUNNEL_HOSTNAME": "", "CODING_API_KEY": secrets.token_hex(32),
              "POCKET_ID_ENCRYPTION_KEY": secrets.token_hex(32),
              "IMAGE_COOKIE_SECRET": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
              "OIDC_CLIENT_ID": "", "OIDC_CLIENT_SECRET": "", "CODING_BIND": os.environ["CODING_BIND"]}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write("# Private cook-4090 credentials. Do not commit or share.\n")
        output.writelines(f"{key}={value}\n" for key, value in values.items())
    print("Created private .env. Set FUNNEL_HOSTNAME after Tailscale enrollment.")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action != "init":
        refresh_docker_group("src.stack")
    env_file = ROOT / ".env"
    if action == "init":
        initialize(env_file)
        return
    if not shutil.which("docker"):
        raise ValueError("Docker with Compose is required. No system packages were changed.")
    config = load_env(env_file)
    env = dict(os.environ, **config)
    env["CODING_BIND"] = validate_bind(env.get("CODING_BIND", "127.0.0.1"))
    env.update(STACK_UID=str(os.getuid()), STACK_GID=str(os.getgid()))
    # Enrollment starts only tailscaled. Compose still parses every service.
    if action in {"prepare", "enroll", "status", "down", "unpublish"} and not env.get("FUNNEL_HOSTNAME"):
        env["FUNNEL_HOSTNAME"] = "pending.invalid"
    if not re.fullmatch(r"[a-z0-9-]+\.[a-z0-9.-]+\.ts\.net", env.get("FUNNEL_HOSTNAME", "")) and env.get("FUNNEL_HOSTNAME") != "pending.invalid":
        raise ValueError("Set FUNNEL_HOSTNAME to this node's full .ts.net name in .env")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", env.get("CODING_API_KEY", "")):
        raise ValueError("CODING_API_KEY must have at least 32 letters, digits, underscores, or hyphens")
    if action in {"up", "publish"} and not all(config.get(k) for k in ("OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET")):
        raise ValueError("Enroll your passkey and configure the Pocket ID client before starting image access")
    compose = ["docker", "compose", "--project-directory", str(ROOT / "config"), "--env-file", str(env_file),
               "-f", str(ROOT / "config/compose.yaml"), "-f", str(ROOT / "config/compose.passkeys.yaml")]

    def run(*args):
        subprocess.run([*compose, *args], cwd=ROOT, env=env, check=True)

    def tailscale(*args):
        run("exec", "-T", "tailscale", "tailscale", *args)

    (ROOT / "build/stack-ipc").mkdir(parents=True, exist_ok=True)
    if action == "prepare":
        run("pull", "tailscale", "caddy", "pocket-id", "image-auth")
        run("build", "image-web")
    elif action == "enroll":
        run("up", "-d", "--wait", "tailscale")
        tailscale("up", "--hostname=cook-4090", "--accept-dns=false")
        tailscale("status")
        print("Set FUNNEL_HOSTNAME in .env to the assigned cook-4090 node name.")
    elif action == "identity":
        run("up", "-d", "tailscale", "pocket-id", "caddy")
        tailscale("serve", "--bg", "--tcp=8443", "tcp://127.0.0.1:9443")
        print(f"Enroll your separate passkey at https://{env['FUNNEL_HOSTNAME']}:8443/setup from a tailnet device.")
    elif action == "up":
        run("up", "-d", "--build")
        tailscale("serve", "--bg", "--tcp=8000", "tcp://127.0.0.1:8000")
        import json
        status = subprocess.check_output([*compose, "exec", "-T", "tailscale", "tailscale", "serve", "status", "--json"],
                                         cwd=ROOT, env=env, text=True)
        forwarding = json.loads(status)
        if "443" not in (forwarding.get("TCP") or {}):
            tailscale("serve", "--bg", "--tcp=443", "tcp://127.0.0.1:8443")
        print(f"Coding API: http://{env['FUNNEL_HOSTNAME']}:8000/v1")
        print(f"Local API: http://{env['CODING_BIND']}:8000/v1")
        print(f"Images: https://{env['FUNNEL_HOSTNAME']} (existing publication state is preserved)")
    elif action == "publish":
        # Only these TLS listeners become public. Coding never uses Funnel.
        tailscale("serve", "--bg", "--tcp=443", "off")
        tailscale("serve", "--bg", "--tcp=8443", "off")
        tailscale("funnel", "--bg", "--tcp=443", "tcp://127.0.0.1:8443")
        tailscale("funnel", "--bg", "--tcp=8443", "tcp://127.0.0.1:9443")
        print(f"Public images with passkeys: https://{env['FUNNEL_HOSTNAME']}")
    elif action == "unpublish":
        tailscale("funnel", "--bg", "--tcp=443", "off")
        tailscale("funnel", "--bg", "--tcp=8443", "off")
        tailscale("serve", "--bg", "--tcp=443", "tcp://127.0.0.1:8443")
        tailscale("serve", "--bg", "--tcp=8443", "tcp://127.0.0.1:9443")
    elif action == "check":
        run("config", "--quiet")
        run("run", "--rm", "--no-deps", "caddy", "caddy", "validate", "--config", "/etc/caddy/Caddyfile")
    elif action == "status":
        run("ps")
        tailscale("status")
        tailscale("serve", "status")
        tailscale("funnel", "status")
    elif action == "down":
        run("down")
    else:
        raise ValueError("Choose init, prepare, enroll, identity, up, publish, unpublish, check, status, or down")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
