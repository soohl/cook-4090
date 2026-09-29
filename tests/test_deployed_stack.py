"""Opt-in checks against this checkout's running private Docker stack."""

import json
import os
from pathlib import Path
import subprocess
import socket
import ssl
import unittest

import httpx

from src import ROOT
from src.stack import load_env


@unittest.skipUnless(os.environ.get("COOK_STACK_LIVE") == "1", "Set COOK_STACK_LIVE=1 for deployed checks")
class DeployedStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_env(ROOT / ".env")
        cls.client = httpx.Client(trust_env=False, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_private_api_key_and_route_isolation(self):
        for address in (self.config["CODING_BIND"], self.config["FUNNEL_HOSTNAME"]):
            base = f"http://{address}:8000"
            with self.subTest(address=address):
                self.assertEqual(self.client.get(base + "/v1/models").status_code, 401)
                for auth in ({"Authorization": "Bearer " + self.config["CODING_API_KEY"]},
                             {"x-api-key": self.config["CODING_API_KEY"]}):
                    models = self.client.get(base + "/v1/models", headers=auth)
                    self.assertEqual(models.status_code, 200)
                    self.assertEqual(models.json()["data"][0]["id"], os.environ["MODEL_ID"])
                    self.assertEqual(self.client.get(base + "/images/options", headers=auth).status_code, 404)

    def test_images_and_assets_require_login(self):
        base = "https://" + self.config["FUNNEL_HOSTNAME"]
        for path in ("/", "/api/sessions", "/api/options", "/api/health", "/v1/models"):
            with self.subTest(path=path):
                response = self.client.get(base + path)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.headers["location"].startswith(base + ":8443/authorize?"))

    def test_tls_and_tunnel_preserve_browser_to_host_encryption(self):
        hostname = self.config['FUNNEL_HOSTNAME']
        for port in (443, 8443):
            with self.subTest(port=port):
                context = ssl.create_default_context()
                context.minimum_version = ssl.TLSVersion.TLSv1_3
                with socket.create_connection((hostname, port), timeout=10) as raw:
                    with context.wrap_socket(raw, server_hostname=hostname) as connection:
                        self.assertEqual(connection.version(), 'TLSv1.3')
                context.maximum_version = ssl.TLSVersion.TLSv1_2
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                with socket.create_connection((hostname, port), timeout=10) as raw:
                    with self.assertRaises(ssl.SSLError):
                        context.wrap_socket(raw, server_hostname=hostname)
        config = json.loads(subprocess.check_output([
            'docker', 'exec', 'cook-4090-tailscale-1', 'tailscale', 'serve', 'status', '--json']))
        self.assertFalse(config.get('Web'))
        self.assertEqual(config['TCP']['443'], {'TCPForward': '127.0.0.1:8443'})
        self.assertEqual(config['TCP']['8443'], {'TCPForward': '127.0.0.1:9443'})
        self.assertEqual(config['TCP']['8000'], {'TCPForward': '127.0.0.1:8000'})
        self.assertTrue(all(key.rsplit(':', 1)[-1] in {'443', '8443'}
                            for key, enabled in config.get('AllowFunnel', {}).items() if enabled))

    def test_unix_socket_and_identity_volume_permissions(self):
        self.assertEqual((ROOT / "build/stack-ipc/inference.sock").stat().st_mode & 0o777, 0o600)
        command = ["docker", "exec", "cook-4090-image-web-1", "python", "-c",
                   'import httpx; c=httpx.Client(transport=httpx.HTTPTransport(uds="/inference/inference.sock"),'
                   'base_url="http://inference"); assert c.get("/health").status_code==200']
        subprocess.run(command, check=True, capture_output=True)
        permissions = subprocess.check_output(["docker", "exec", "cook-4090-pocket-id-1",
                                               "stat", "-c", "%u:%g:%a", "/app/data"], text=True).strip()
        self.assertEqual(permissions, "1000:1000:700")

    def test_only_coding_has_a_host_port_and_apps_are_internal(self):
        names = ["tailscale", "caddy", "pocket-id", "image-auth", "image-web"]
        containers = json.loads(subprocess.check_output(
            ["docker", "inspect", *[f"cook-4090-{name}-1" for name in names]], text=True))
        for name, container in zip(names, containers):
            self.assertTrue(container["State"]["Running"], name)
            logs = container['HostConfig']['LogConfig']
            self.assertEqual(logs['Type'], 'local')
            self.assertEqual(logs['Config']['max-size'], os.environ['STACK_LOG_MAX_SIZE'])
            self.assertEqual(logs['Config']['max-file'], os.environ['STACK_LOG_MAX_FILES'])
            if name == "image-web":
                saved = [m for m in container["Mounts"] if m["Destination"] == "/data/images"]
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["Type"], "volume")
                self.assertTrue(saved[0]["RW"])
            if name in {"image-web", "pocket-id"}:
                self.assertEqual(container["State"]["Health"]["Status"], "healthy", name)
            ports = container["HostConfig"]["PortBindings"] or {}
            if name == "tailscale":
                self.assertEqual(ports, {"8000/tcp": [{"HostIp": self.config["CODING_BIND"], "HostPort": "8000"}]})
            else:
                self.assertEqual(ports, {}, name)
            if name in {"pocket-id", "image-auth", "image-web"}:
                self.assertEqual(set(container["NetworkSettings"]["Networks"]), {"cook-4090_application"})
        network = json.loads(subprocess.check_output(["docker", "network", "inspect", "cook-4090_application"], text=True))[0]
        self.assertTrue(network["Internal"])
