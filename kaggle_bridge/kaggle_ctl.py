#!/usr/bin/env python3
"""
Local Controller CLI for Antigravity - Kaggle GPU Bridge (kaggle_ctl.py)
------------------------------------------------------------------------
Commands:
  python kaggle_ctl.py up                  -> Push daemon kernel to Kaggle 2x T4 & wait for live tunnel URL
  python kaggle_ctl.py status              -> Show live GPU status, jobs, & remaining 30h quota
  python kaggle_ctl.py bash "<cmd>"        -> Run bash command on Kaggle GPU node
  python kaggle_ctl.py bash_bg "<cmd>"     -> Start background bash job on Kaggle GPU node
  python kaggle_ctl.py py "<code|@file>"   -> Run Python code in persistent GPU VRAM worker
  python kaggle_ctl.py setup               -> Trigger automated scMamba2 (torch 2.3 + mamba-ssm 2.2.2) setup
  python kaggle_ctl.py upload <src> <dst>  -> Upload local file to Kaggle
  python kaggle_ctl.py download <src> <dst>-> Download file from Kaggle to local PC
  python kaggle_ctl.py down                -> Immediately shut down Kaggle GPU kernel (0 idle quota waste!)
"""

import base64
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
KERNEL_DIR = BASE_DIR / "kernel"
STAGING_KERNEL_DIR = BASE_DIR / "_staging_kernel"
STATE_FILE = BASE_DIR / "active_tunnel.json"
ENV_FILE = PROJECT_ROOT / ".env"


def load_dotenv(env_path: Path) -> dict[str, str]:
    """Loads key=value pairs from .env file into os.environ without external dependencies."""
    loaded = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            loaded[k] = v
            if k not in os.environ:
                os.environ[k] = v
    return loaded


load_dotenv(ENV_FILE)

AUTH_TOKEN = os.environ.get("AG_BRIDGE_AUTH_TOKEN", "")
NTFY_TOPIC = os.environ.get("AG_BRIDGE_NTFY_TOPIC", "")
KERNEL_ID = os.environ.get("KAGGLE_KERNEL_ID", "tughankovucu/antigravity-gpu-bridge")
IDLE_TIMEOUT_SEC = os.environ.get("AG_IDLE_TIMEOUT_SEC", "1200")

SSL_CTX = ssl._create_unverified_context()


def ensure_env_configured():
    if not AUTH_TOKEN or not NTFY_TOPIC:
        raise RuntimeError(
            f"Missing AG_BRIDGE_AUTH_TOKEN or AG_BRIDGE_NTFY_TOPIC. "
            f"Please copy `.env.example` to `.env` in {PROJECT_ROOT} and fill in your credentials."
        )
    # Also ensure ~/.kaggle/access_token is synced if KAGGLE_API_TOKEN is provided in .env
    api_tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if api_tok:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        tok_file = kaggle_dir / "access_token"
        if not tok_file.exists() or tok_file.read_text(encoding="utf-8").strip() != api_tok:
            tok_file.write_text(api_tok, encoding="utf-8")


def load_tunnel_url() -> str:
    if not STATE_FILE.exists():
        raise RuntimeError("No active tunnel found. Run `python kaggle_ctl.py up` first.")
    data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return data["tunnel_url"]


def api_call(method: str, endpoint: str, payload: dict = None, timeout: int = 300) -> dict:
    url = load_tunnel_url().rstrip("/") + endpoint
    headers = {
        "X-Auth-Token": AUTH_TOKEN,
        "Content-Type": "application/json",
        "User-Agent": "Antigravity-Kaggle-Client/1.0",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_kaggle_cli(args: list[str]) -> str:
    ensure_env_configured()
    env = os.environ.copy()
    py_scripts = str(Path(sys.executable).parent / "Scripts")
    env["PATH"] = py_scripts + os.pathsep + env.get("PATH", "")
    kaggle_exe = Path(py_scripts) / "kaggle.exe"
    kaggle_bin = str(kaggle_exe) if kaggle_exe.exists() else (shutil.which("kaggle.exe") or "kaggle")
    p = subprocess.run([kaggle_bin] + args, capture_output=True, text=True, env=env)
    return (p.stdout or "") + (p.stderr or "")


def cmd_up():
    ensure_env_configured()
    print("[1/3] Pushing Antigravity GPU Daemon kernel to Kaggle (2x Tesla T4)...")
    push_time = time.time()
    # Prepare temporary git-ignored staging kernel with .env tokens injected
    if STAGING_KERNEL_DIR.exists():
        shutil.rmtree(STAGING_KERNEL_DIR, ignore_errors=True)
    shutil.copytree(KERNEL_DIR, STAGING_KERNEL_DIR)
    try:
        daemon_py = STAGING_KERNEL_DIR / "agent_daemon.py"
        code_txt = daemon_py.read_text(encoding="utf-8")
        code_txt = code_txt.replace("__AG_BRIDGE_AUTH_TOKEN__", AUTH_TOKEN)
        code_txt = code_txt.replace("__AG_BRIDGE_NTFY_TOPIC__", NTFY_TOPIC)
        daemon_py.write_text(code_txt, encoding="utf-8")

        meta_json = STAGING_KERNEL_DIR / "kernel-metadata.json"
        meta_obj = json.loads(meta_json.read_text(encoding="utf-8"))
        meta_obj["id"] = KERNEL_ID
        meta_json.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

        out = run_kaggle_cli(["kernels", "push", "-p", str(STAGING_KERNEL_DIR)])
        print(out.strip())
    finally:
        if STAGING_KERNEL_DIR.exists():
            shutil.rmtree(STAGING_KERNEL_DIR, ignore_errors=True)

    print("[2/3] Waiting for Kaggle 2x T4 GPU VM to boot and publish Cloudflare tunnel URL via ntfy.sh...")
    start_wait = time.time()
    seen_msg_ids = set()
    while time.time() - start_wait < 240:
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}/json?poll=1&since={int(push_time - 5)}",
                headers={"User-Agent": "Antigravity-Kaggle-Client/1.0"},
            )
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as r:
                lines = [ln.strip() for ln in r.read().decode("utf-8").splitlines() if ln.strip()]
            for ln in reversed(lines):
                ev = json.loads(ln)
                msg_id = ev.get("id")
                if ev.get("event") == "message":
                    try:
                        msg_obj = json.loads(ev.get("message", "{}"))
                    except Exception:
                        msg_obj = {}
                    
                    if msg_id and msg_id not in seen_msg_ids:
                        seen_msg_ids.add(msg_id)
                        evt_name = msg_obj.get("event", "status")
                        evt_stat = msg_obj.get("status") or msg_obj.get("error") or ""
                        print(f"  [ntfy -> {int(time.time() - start_wait)}s] {evt_name.upper()}: {evt_stat}", flush=True)

                    t_url = msg_obj.get("tunnel_url")
                    if t_url and msg_obj.get("timestamp", 0) >= push_time - 10:
                        STATE_FILE.write_text(json.dumps(msg_obj, indent=2), encoding="utf-8")
                        print(f"\n[3/3] TUNNEL CONNECTED: {t_url}")
                        for _ in range(10):
                            try:
                                st = api_call("GET", "/status", timeout=15)
                                print("\n=== LIVE KAGGLE GPU SERVER STATUS ===")
                                print(json.dumps(st, indent=2, ensure_ascii=False))
                                return
                            except Exception:
                                time.sleep(2)
        except Exception:
            pass
        k_status = run_kaggle_cli(["kernels", "status", KERNEL_ID]).strip()
        print(f"  [{int(time.time() - start_wait)}s] {k_status}", flush=True)
        time.sleep(5)

    raise TimeoutError("Timed out waiting for Kaggle GPU tunnel after 240 seconds.")



def cmd_status():
    print("--- Kaggle Quota ---")
    print(run_kaggle_cli(["quota"]).strip())
    print("\n--- Kernel Status ---")
    print(run_kaggle_cli(["kernels", "status", KERNEL_ID]).strip())
    if STATE_FILE.exists():
        print("\n--- Live Tunnel Status ---")
        try:
            st = api_call("GET", "/status", timeout=15)
            print(json.dumps(st, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Tunnel unreachable ({e})")


def cmd_bash(cmd: str, bg: bool = False):
    res = api_call("POST", "/exec_bash", {"cmd": cmd, "bg": bg, "timeout": 600}, timeout=610)
    if bg:
        print(json.dumps(res, indent=2))
    else:
        if res.get("stdout"):
            print(res["stdout"], end="")
        if res.get("stderr"):
            print(res["stderr"], file=sys.stderr, end="")
        print(f"\n[Exit code: {res.get('returncode')} | {res.get('elapsed_sec')}s]")


def cmd_py(arg: str):
    if arg.startswith("@"):
        code = Path(arg[1:]).read_text(encoding="utf-8")
    else:
        code = arg
    res = api_call("POST", "/exec_python", {"code": code, "timeout": 1800}, timeout=1810)
    if res.get("stdout"):
        print(res["stdout"], end="")
    if res.get("stderr"):
        print(res["stderr"], file=sys.stderr, end="")
    print(f"\n[Python OK: {res.get('ok')} | {res.get('elapsed_sec')}s]")


def cmd_setup():
    res = api_call("POST", "/setup_scmamba")
    print("Triggered scMamba2 setup:", res)
    while True:
        time.sleep(5)
        st = api_call("GET", "/status")
        status = st.get("setup_status")
        tail = st.get("setup_log_tail", "").splitlines()
        last_line = tail[-1] if tail else ""
        print(f"[{status}] {last_line}", flush=True)
        if status in ("completed", "failed"):
            print("\n=== FULL SETUP LOG TAIL ===")
            print(st.get("setup_log_tail", ""))
            break


def cmd_upload(local_path: str, remote_path: str):
    raw = Path(local_path).read_bytes()
    chunk_size = 6 * 1024 * 1024  # 6 MB chunks for reliable Cloudflare tunnel transport
    if len(raw) <= chunk_size:
        b64 = base64.b64encode(raw).decode("ascii")
        res = api_call("POST", "/upload", {"path": remote_path, "content_b64": b64, "append": False}, timeout=300)
        print(json.dumps(res, indent=2))
    else:
        total = len(raw)
        offset = 0
        part = 0
        while offset < total:
            chunk = raw[offset : offset + chunk_size]
            b64 = base64.b64encode(chunk).decode("ascii")
            res = api_call(
                "POST",
                "/upload",
                {"path": remote_path, "content_b64": b64, "append": (part > 0)},
                timeout=300,
            )
            offset += len(chunk)
            part += 1
            print(f"  [Chunk {part}] Uploaded {offset}/{total} bytes ({offset * 100.0 / total:.1f}%) -> {remote_path}", flush=True)
        print(json.dumps(res, indent=2))


def cmd_download(remote_path: str, local_path: str):
    res = api_call("POST", "/download", {"path": remote_path}, timeout=300)
    if not res.get("ok"):
        print("Error:", res, file=sys.stderr)
        sys.exit(1)
    raw = base64.b64decode(res["content_b64"])
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_bytes(raw)
    print(f"Downloaded {len(raw)} bytes to {local_path}")


def cmd_down():
    try:
        res = api_call("POST", "/shutdown", timeout=15)
        print(res)
    except Exception as e:
        print(f"Shutdown signal sent ({e})")
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    time.sleep(5)
    print(run_kaggle_cli(["kernels", "status", KERNEL_ID]).strip())


def interactive_menu():
    while True:
        print("\n" + "=" * 60)
        print("   ANTIGRAVITY - KAGGLE 2x T4 GPU KONTROL PANELİ")
        print("=" * 60)
        print("  [1] GPU Sunucusunu Başlat (up)")
        print("  [2] Durum ve Kalan 30 Saatlik Kotayı Gör (status)")
        print("  [3] scMamba2 Ortamını Otomatik Kur (setup)")
        print("  [4] Canlı Linux (Bash) Komutu Çalıştır")
        print("  [5] Canlı GPU Python Kodu / Dosyası Çalıştır (Kalıcı VRAM)")
        print("  [6] GPU Sunucusunu Hemen Kapat - Kotayı Durdur (down)")
        print("  [0] Çıkış")
        print("=" * 60)
        choice = input("Seçiminiz (0-6): ").strip()
        try:
            if choice == "1":
                cmd_up()
            elif choice == "2":
                cmd_status()
            elif choice == "3":
                cmd_setup()
            elif choice == "4":
                c = input("Bash komutu (ör. nvidia-smi): ").strip()
                if c:
                    cmd_bash(c, bg=False)
            elif choice == "5":
                c = input("Python kodu veya @dosya.py: ").strip()
                if c:
                    cmd_py(c)
            elif choice == "6":
                cmd_down()
            elif choice == "0":
                break
        except Exception as e:
            print(f"\n[HATA] {e}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "menu":
        interactive_menu()
        return
    action = sys.argv[1]
    if action == "up":
        cmd_up()
    elif action == "status":
        cmd_status()
    elif action == "bash":
        cmd_bash(sys.argv[2], bg=False)
    elif action == "bash_bg":
        cmd_bash(sys.argv[2], bg=True)
    elif action == "py":
        cmd_py(sys.argv[2])
    elif action == "setup":
        cmd_setup()
    elif action == "upload":
        cmd_upload(sys.argv[2], sys.argv[3])
    elif action == "download":
        cmd_download(sys.argv[2], sys.argv[3])
    elif action == "down":
        cmd_down()
    else:
        print(f"Unknown command: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
