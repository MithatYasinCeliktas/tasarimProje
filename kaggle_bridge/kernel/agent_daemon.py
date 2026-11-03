#!/usr/bin/env python3
"""
Antigravity - Kaggle God-Mode GPU Daemon (agent_daemon.py)
----------------------------------------------------------
Runs inside Kaggle GPU container (2x Tesla T4).
Features:
1. Zero-dependency HTTP/RPC Server (ThreadingHTTPServer on 127.0.0.1:8765)
2. Persistent Stateful GPU Python Worker Subprocess (isolated from daemon so torch can be
   installed/upgraded without os._exit(0) killing the Kaggle session!)
3. Synchronous & Background Bash execution with live log streaming
4. Binary File Upload / Download (/kaggle/working <-> Local Windows PC)
5. Outbound Cloudflare Quick Tunnel (trycloudflare.com) + ntfy.sh Encrypted Rendezvous
6. Dead-Man's Switch Watchdog: Automatically exits (stopping Kaggle 30h GPU quota clock)
   if idle for > IDLE_TIMEOUT_SEC with no active background jobs.
"""

import base64
import glob
import io
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Configuration (injected from local .env during `kaggle_ctl.py up` staging)
DAEMON_PORT = 8765
WORKER_PORT = 8766
AUTH_TOKEN = os.environ.get("AG_BRIDGE_AUTH_TOKEN", "__AG_BRIDGE_AUTH_TOKEN__")
NTFY_TOPIC = os.environ.get("AG_BRIDGE_NTFY_TOPIC", "__AG_BRIDGE_NTFY_TOPIC__")
IDLE_TIMEOUT_SEC = int(os.environ.get("AG_IDLE_TIMEOUT_SEC", "1200"))

STATE = {
    "boot_time": time.time(),
    "last_activity": time.time(),
    "tunnel_url": None,
    "setup_status": "not_started",
    "setup_log": "",
    "bg_jobs": {},  # job_id -> {"proc": Popen, "cmd": str, "log_path": str, "start_time": float}
    "shutdown_requested": False,
}

WORKER_PROC = None
WORKER_LOCK = threading.Lock()

# Embedded code for the Persistent Stateful GPU Python Worker
WORKER_CODE = r'''
import io
import json
import socket
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/input/datasets/tughankovucu/sc-mamba2-model")

GLOBAL_NS = {"__name__": "__main__"}

def handle_client(conn):
    try:
        data = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        req = json.loads(data.decode("utf-8"))
        code = req.get("code", "")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        t0 = time.time()
        ok = True
        result_repr = None
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Support expression evaluation on last line if possible
                exec(compile(code, "<antigravity_cell>", "exec"), GLOBAL_NS)
        except Exception:
            ok = False
            stderr_buf.write(traceback.format_exc())
        dt = time.time() - t0
        resp = {
            "ok": ok,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "elapsed_sec": round(dt, 4),
        }
        conn.sendall(json.dumps(resp).encode("utf-8"))
    except Exception as e:
        try:
            conn.sendall(json.dumps({"ok": False, "stdout": "", "stderr": str(e), "elapsed_sec": 0}).encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", int(sys.argv[1])))
    srv.listen(5)
    while True:
        conn, _ = srv.accept()
        handle_client(conn)

if __name__ == "__main__":
    main()
'''


def touch_activity():
    STATE["last_activity"] = time.time()


def has_active_bg_jobs():
    for jid, info in list(STATE["bg_jobs"].items()):
        if info["proc"].poll() is None:
            return True
    if STATE["setup_status"] == "running":
        return True
    return False


def ensure_worker_started():
    global WORKER_PROC
    with WORKER_LOCK:
        if WORKER_PROC is not None and WORKER_PROC.poll() is None:
            return
        worker_script = "/tmp/ag_gpu_worker.py"
        with open(worker_script, "w", encoding="utf-8") as f:
            f.write(WORKER_CODE)
        WORKER_PROC = subprocess.Popen(
            [sys.executable, worker_script, str(WORKER_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until worker port is listening
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", WORKER_PORT), timeout=0.5):
                    return
            except Exception:
                time.sleep(0.2)


def restart_worker():
    global WORKER_PROC
    with WORKER_LOCK:
        if WORKER_PROC is not None and WORKER_PROC.poll() is None:
            WORKER_PROC.kill()
            WORKER_PROC.wait()
        WORKER_PROC = None
    ensure_worker_started()


def exec_in_worker(code: str, timeout: int = 3600) -> dict:
    ensure_worker_started()
    with socket.create_connection(("127.0.0.1", WORKER_PORT), timeout=5) as s:
        s.settimeout(timeout)
        payload = json.dumps({"code": code}).encode("utf-8")
        s.sendall(payload)
        s.shutdown(socket.SHUT_WR)
        resp_bytes = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            resp_bytes += chunk
    return json.loads(resp_bytes.decode("utf-8"))


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception as e:
        return f"nvidia-smi error: {e}"


def run_scmamba_setup():
    """
    Sets up the exact working stack for scMamba2 + BiMamba:
    - Reuses cached CUDA 12.1 nvidia-* wheels from sc-mamba2-wheels where possible
    - Installs torch==2.3.0+cu121, torchvision==0.18.0+cu121, torchaudio==2.3.0+cu121
    - Installs torchtext==0.18.0 (--no-deps)
    - Installs prebuilt torch2.3 wheels for causal-conv1d==1.4.0 and mamba-ssm==2.2.2
    - Installs transformers==4.43.4 and scgpt (--no-deps)
    - Copies and patches scmamba2 repo in /kaggle/working/scmamba2
    - Restarts the GPU worker subprocess so the new torch 2.3.0 stack is loaded cleanly!
    """
    STATE["setup_status"] = "running"
    log_lines = []

    def log(msg):
        print(f"[SETUP] {msg}", flush=True)
        log_lines.append(msg)
        STATE["setup_log"] = "\n".join(log_lines[-200:])
        touch_activity()

    try:
        log("Step 1/5: Preparing local wheel cache from sc-mamba2-wheels...")
        wheel_src_candidates = glob.glob("/kaggle/input/**/sc-mamba2-wheels", recursive=True)
        find_links_args = []
        if wheel_src_candidates:
            wdir = wheel_src_candidates[0]
            log(f"Found wheel dataset at {wdir}")
            find_links_args = ["--find-links", wdir]

        def run_pip(args_list):
            cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location"] + args_list
            log("Running: " + " ".join(cmd))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                log(line.rstrip())
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"pip command failed with code {p.returncode}")

        log("Step 2/5: Installing torch==2.3.0+cu121 stack...")
        run_pip(
            find_links_args
            + [
                "torch==2.3.0",
                "torchvision==0.18.0",
                "torchaudio==2.3.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )

        log("Step 3/5: Installing torchtext==0.18.0, causal-conv1d 1.4.0, and mamba-ssm 2.2.2 (torch2.3 wheels)...")
        run_pip(["--no-deps", "torchtext==0.18.0"])
        run_pip(
            [
                "--force-reinstall",
                "--no-deps",
                "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
                "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
            ]
        )

        log("Step 4/5: Installing transformers==4.43.4 and scgpt (--no-deps)...")
        run_pip(["transformers==4.43.4"])
        run_pip(["--no-deps", "scgpt"])

        log("Step 5/5: Copying and patching scmamba2 source into /kaggle/working/scmamba2...")
        model_dirs = [d for d in glob.glob("/kaggle/input/**/scmamba2", recursive=True) if os.path.isdir(d)]
        if model_dirs:
            src_scmamba = model_dirs[0]
            dst_scmamba = "/kaggle/working/scmamba2"
            shutil.copytree(src_scmamba, dst_scmamba, dirs_exist_ok=True)
            subprocess.run(
                ["find", dst_scmamba, "-name", "*.py", "-exec", "sed", "-i", "-E", r"s/\bscmamba\b/scmamba2/g", "{}", "+"],
                check=True,
            )
            log(f"Patched scmamba2 in {dst_scmamba}")
        else:
            log("WARNING: sc-mamba2-model/scmamba2 not found in /kaggle/input")

        log("Restarting GPU worker subprocess to load clean torch 2.3.0 + mamba-ssm 2.2.2...")
        restart_worker()
        STATE["setup_status"] = "completed"
        log("SETUP COMPLETED SUCCESSFULLY!")
    except Exception as e:
        STATE["setup_status"] = "failed"
        log(f"SETUP FAILED: {e}\n{traceback.format_exc()}")


class AgentHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # keep daemon stdout clean

    def _check_auth(self):
        tok = self.headers.get("X-Auth-Token", "")
        if tok != AUTH_TOKEN:
            self._send_json(401, {"error": "Unauthorized"})
            return False
        touch_activity()
        return True

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if not self._check_auth():
            return
        if self.path == "/status":
            jobs_summary = {}
            for jid, info in STATE["bg_jobs"].items():
                rc = info["proc"].poll()
                tail = ""
                if os.path.exists(info["log_path"]):
                    with open(info["log_path"], "r", encoding="utf-8", errors="replace") as f:
                        tail = f.read()[-2000:]
                jobs_summary[jid] = {
                    "cmd": info["cmd"],
                    "running": rc is None,
                    "returncode": rc,
                    "elapsed_sec": round(time.time() - info["start_time"], 1),
                    "log_tail": tail,
                }
            self._send_json(
                200,
                {
                    "ok": True,
                    "uptime_sec": round(time.time() - STATE["boot_time"], 1),
                    "idle_sec": round(time.time() - STATE["last_activity"], 1),
                    "idle_timeout_sec": IDLE_TIMEOUT_SEC,
                    "tunnel_url": STATE["tunnel_url"],
                    "gpu": get_gpu_info(),
                    "setup_status": STATE["setup_status"],
                    "setup_log_tail": STATE["setup_log"][-1500:],
                    "bg_jobs": jobs_summary,
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._check_auth():
            return
        req = self._read_json()

        if self.path == "/exec_bash":
            cmd = req.get("cmd", "")
            cwd = req.get("cwd", "/kaggle/working")
            timeout = req.get("timeout", 300)
            bg = req.get("bg", False)
            if bg:
                jid = f"job_{int(time.time()*1000)}"
                log_path = f"/tmp/{jid}.log"
                log_f = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                STATE["bg_jobs"][jid] = {
                    "proc": proc,
                    "cmd": cmd,
                    "log_path": log_path,
                    "start_time": time.time(),
                }
                self._send_json(200, {"ok": True, "bg": True, "job_id": jid, "pid": proc.pid, "log_path": log_path})
            else:
                t0 = time.time()
                try:
                    p = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=cwd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    self._send_json(
                        200,
                        {
                            "ok": p.returncode == 0,
                            "returncode": p.returncode,
                            "stdout": p.stdout,
                            "stderr": p.stderr,
                            "elapsed_sec": round(time.time() - t0, 3),
                        },
                    )
                except subprocess.TimeoutExpired as e:
                    self._send_json(
                        200,
                        {
                            "ok": False,
                            "returncode": -1,
                            "stdout": e.stdout or "",
                            "stderr": f"TimeoutExpired after {timeout}s\n" + (e.stderr or ""),
                            "elapsed_sec": round(time.time() - t0, 3),
                        },
                    )

        elif self.path == "/exec_python":
            code = req.get("code", "")
            timeout = req.get("timeout", 1800)
            res = exec_in_worker(code, timeout=timeout)
            self._send_json(200, res)

        elif self.path == "/restart_worker":
            restart_worker()
            self._send_json(200, {"ok": True, "message": "GPU worker restarted cleanly"})

        elif self.path == "/setup_scmamba":
            if STATE["setup_status"] != "running":
                threading.Thread(target=run_scmamba_setup, daemon=True).start()
            self._send_json(200, {"ok": True, "setup_status": STATE["setup_status"]})

        elif self.path == "/upload":
            dest = req.get("path", "/kaggle/working/uploaded_file")
            b64 = req.get("content_b64", "")
            append = bool(req.get("append", False))
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            raw = base64.b64decode(b64)
            mode = "ab" if append else "wb"
            with open(dest, mode) as f:
                f.write(raw)
            self._send_json(200, {"ok": True, "path": dest, "bytes": len(raw), "total_bytes": os.path.getsize(dest)})

        elif self.path == "/download":
            src = req.get("path", "")
            if not os.path.exists(src):
                self._send_json(404, {"ok": False, "error": f"File not found: {src}"})
                return
            with open(src, "rb") as f:
                raw = f.read()
            self._send_json(
                200,
                {
                    "ok": True,
                    "path": src,
                    "bytes": len(raw),
                    "content_b64": base64.b64encode(raw).decode("ascii"),
                },
            )

        elif self.path == "/set_timeout":
            global IDLE_TIMEOUT_SEC
            IDLE_TIMEOUT_SEC = int(req.get("idle_timeout_sec", 1200))
            self._send_json(200, {"ok": True, "idle_timeout_sec": IDLE_TIMEOUT_SEC})

        elif self.path == "/shutdown":
            STATE["shutdown_requested"] = True
            self._send_json(200, {"ok": True, "message": "Shutting down Kaggle GPU container immediately..."})
            def _exit_soon():
                time.sleep(1.0)
                os._exit(0)
            threading.Thread(target=_exit_soon, daemon=True).start()

        else:
            self._send_json(404, {"error": "Unknown endpoint"})


def send_ntfy(title: str, payload_dict: dict) -> bool:
    payload = json.dumps(payload_dict).encode("utf-8")
    ctx = ssl._create_unverified_context()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=payload,
                headers={"Content-Type": "application/json", "Title": title},
                method="POST",
            )
            with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
                print(f"[NTFY] {title} sent: {payload_dict.get('event')}", flush=True)
                return True
        except Exception as e:
            print(f"[NTFY] Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(1)
    return False


def publish_rendezvous(tunnel_url: str):
    payload = {
        "event": "ready",
        "tunnel_url": tunnel_url,
        "gpu": get_gpu_info(),
        "timestamp": time.time(),
    }
    send_ntfy("Kaggle-GPU-Bridge-Ready", payload)


def start_cloudflared():
    try:
        cf_bin = "/tmp/cloudflared"
        if not os.path.exists(cf_bin):
            print("[TUNNEL] Downloading cloudflared binary...", flush=True)
            send_ntfy("Kaggle-GPU-Status", {"event": "downloading_cf", "status": "Downloading cloudflared binary...", "timestamp": time.time()})
            p = subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--output",
                    cf_bin,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                ],
                capture_output=True,
                text=True,
            )
            if p.returncode != 0:
                err = f"curl failed ({p.returncode}): {p.stderr}"
                print(f"[TUNNEL] {err}", flush=True)
                send_ntfy("Kaggle-GPU-Error", {"event": "error", "error": err, "timestamp": time.time()})
                with open("/kaggle/working/tunnel_error.txt", "w") as f:
                    f.write(err + "\n")
                return
            os.chmod(cf_bin, 0o755)

        print("[TUNNEL] Starting Cloudflare Quick Tunnel...", flush=True)
        send_ntfy("Kaggle-GPU-Status", {"event": "starting_tunnel", "status": "Cloudflare quick tunnel starting...", "timestamp": time.time()})
        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{DAEMON_PORT}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        url_pattern = re.compile(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)")
        log_tail = []
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            line_s = line.strip()
            print(f"[CF] {line_s}", flush=True)
            log_tail.append(line_s)
            m = url_pattern.search(line)
            if m and STATE["tunnel_url"] is None:
                url = m.group(1)
                STATE["tunnel_url"] = url
                print(f"[TUNNEL] LIVE URL: {url}", flush=True)
                with open("/kaggle/working/tunnel_url.txt", "w") as f:
                    f.write(url + "\n")
                publish_rendezvous(url)

        if STATE["tunnel_url"] is None:
            err = "Cloudflared exited before publishing tunnel: " + "\n".join(log_tail[-10:])
            print(f"[TUNNEL] {err}", flush=True)
            send_ntfy("Kaggle-GPU-Error", {"event": "error", "error": err, "timestamp": time.time()})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[TUNNEL] Exception in start_cloudflared:\n{tb}", flush=True)
        send_ntfy("Kaggle-GPU-Error", {"event": "error", "error": str(e), "traceback": tb, "timestamp": time.time()})


def main():
    print("=== Antigravity Kaggle GPU Daemon Starting ===", flush=True)
    gpu_info = get_gpu_info()
    print(gpu_info, flush=True)

    # Immediate boot ping to ntfy
    send_ntfy("Kaggle-GPU-Booting", {
        "event": "boot",
        "gpu": gpu_info,
        "status": "Container booted, starting RPC daemon...",
        "timestamp": time.time(),
    })

    # Start HTTP RPC server
    server = ThreadingHTTPServer(("127.0.0.1", DAEMON_PORT), AgentHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[SERVER] Listening on 127.0.0.1:{DAEMON_PORT}", flush=True)

    # Start initial worker subprocess
    ensure_worker_started()
    print(f"[WORKER] Persistent Python GPU worker started on 127.0.0.1:{WORKER_PORT}", flush=True)

    # Start cloudflared in background thread
    threading.Thread(target=start_cloudflared, daemon=True).start()

    # Main watchdog loop (protects 30-hour GPU quota!)
    while not STATE["shutdown_requested"]:
        time.sleep(5)
        if not has_active_bg_jobs():
            idle = time.time() - STATE["last_activity"]
            if idle > IDLE_TIMEOUT_SEC:
                print(f"[WATCHDOG] Idle for {idle:.0f}s (> {IDLE_TIMEOUT_SEC}s). Auto-shutting down to save GPU quota!", flush=True)
                os._exit(0)


if __name__ == "__main__":
    main()

