"""One-command local launch of PokerMind: FastAPI backend plus Next.js frontend (M9).

``pokermind launch`` does the following:

1. **Checks the setup.** It needs:
   * Python ≥ 3.12 and the ``[app]`` extra;
   * the frozen research configuration and its CFR hero (the digests are
     verified by the same loader the server uses);
   * Node.js ≥ 20.9 and ``web/node_modules``;
   * free ports.

   Every failure prints the exact command that fixes it.
2. **Builds the web app** when there is no build, when sources changed since
   the last build, or when the API address differs from the one the build was
   made for. The address is baked into the build's ``/api`` proxy.
3. **Starts both services**, each in its own process group, with logs under
   ``runs/launch/``. It then waits until the backend answers ``/health`` and
   the frontend answers both ``/`` and ``/api/health`` (through its proxy).
4. **Prints the URL**, then supervises. On Ctrl-C, SIGTERM, or the exit of
   either service, it stops both process groups: SIGTERM first, then
   SIGKILL. That includes the backend's solver worker processes, so nothing
   is left behind.

macOS and Linux (POSIX process groups). Local use only: it binds 127.0.0.1
by default.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TextIO

MIN_PYTHON: Final = (3, 12)
MIN_NODE: Final = (20, 9)
STARTUP_TIMEOUT: Final = 90.0
STOP_GRACE: Final = 10.0
BUILD_STAMP: Final = "pokermind-launch.json"
WEB_SOURCES: Final = (
    "src",
    "next.config.ts",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
)


class LaunchError(RuntimeError):
    """A problem the user must fix; the message says how."""


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    root: Path
    host: str = "127.0.0.1"
    api_port: int = 8000
    web_port: int = 3000
    config: Path = Path("configs/pokermind-research-v1.json")
    hero: Path | None = None
    recompute_every: int | None = None
    rebuild: bool = False
    open_browser: bool = False

    @property
    def web_dir(self) -> Path:
        return self.root / "web"

    @property
    def log_dir(self) -> Path:
        return self.root / "runs" / "launch"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    @property
    def web_url(self) -> str:
        """The address to open in a browser."""
        return f"http://{'localhost' if self.host == '127.0.0.1' else self.host}:{self.web_port}"

    @property
    def web_probe(self) -> str:
        """The exact address the web server binds (used for health checks)."""
        return f"http://{self.host}:{self.web_port}"


# -- checks -----------------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        raise ValueError(f"no version in {text!r}")
    return tuple(int(g) for g in match.groups() if g is not None)


def check_python() -> None:
    if sys.version_info[:2] < MIN_PYTHON:
        raise LaunchError(f"Python {'.'.join(map(str, MIN_PYTHON))}+ is required")
    missing = [m for m in ("fastapi", "uvicorn") if _missing_module(m)]
    if missing:
        raise LaunchError(
            f"missing Python packages {', '.join(missing)}: run  pip install -e '.[app]'"
        )


def _missing_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def check_research(cfg: LaunchConfig) -> str:
    """Load the frozen configuration and hero exactly as the server will. Returns a summary."""
    from pokermind.app.context import ContextError, load_app_context

    config = cfg.config if cfg.config.is_absolute() else cfg.root / cfg.config
    try:
        ctx = load_app_context(config, cfg.hero)
    except ContextError as exc:
        hint = ""
        if "not found" in str(exc):
            hint = "  (one-time, about 30 minutes; or run  pokermind launch --train-hero)"
        raise LaunchError(f"{exc}{hint}") from exc
    return (
        f"research {ctx.research.version} ({ctx.config_digest[:12]}), hero {ctx.cfr_digest[:12]}, "
        f"exploitability limit {ctx.budget:.4f} chips/hand"
    )


def check_node(cfg: LaunchConfig, run: Callable[..., str] | None = None) -> Path:
    node = shutil.which("node")
    if node is None:
        raise LaunchError("Node.js is not installed (need 20.9 or newer): https://nodejs.org")
    runner = run or (
        lambda *a: subprocess.run(a, capture_output=True, text=True, check=True).stdout
    )
    version = parse_version(runner(node, "--version"))
    if version[:2] < MIN_NODE:
        raise LaunchError(
            f"Node.js {'.'.join(map(str, version))} is too old; need "
            f"{'.'.join(map(str, MIN_NODE))} or newer"
        )
    next_bin = cfg.web_dir / "node_modules" / ".bin" / "next"
    if not next_bin.exists():
        raise LaunchError("web dependencies are not installed: run  (cd web && npm ci)")
    return next_bin


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def check_ports(cfg: LaunchConfig) -> None:
    if cfg.api_port == cfg.web_port:
        raise LaunchError("the API and web ports must differ")
    for name, port, flag in (
        ("API", cfg.api_port, "--api-port"),
        ("web", cfg.web_port, "--web-port"),
    ):
        if not port_free(cfg.host, port):
            raise LaunchError(
                f"port {port} ({name}) is already in use. Stop whatever uses it "
                f"(lsof -nP -iTCP:{port} -sTCP:LISTEN) or choose another with {flag}"
            )


# -- build ------------------------------------------------------------------------------------


def _newest_mtime(paths: Sequence[Path]) -> float:
    newest = 0.0
    for p in paths:
        if p.is_file():
            newest = max(newest, p.stat().st_mtime)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime)
    return newest


def build_reason(cfg: LaunchConfig) -> str | None:
    """Why the web app must be (re)built, or None if the current build is usable."""
    if cfg.rebuild:
        return "rebuild requested"
    build_id = cfg.web_dir / ".next" / "BUILD_ID"
    if not build_id.exists():
        return "no production build yet"
    stamp = cfg.web_dir / ".next" / BUILD_STAMP
    try:
        built_for = json.loads(stamp.read_text()).get("api_url")
    except (OSError, ValueError):
        built_for = None
    if built_for != cfg.api_url:
        return f"the build targets {built_for or 'an unknown API'}, not {cfg.api_url}"
    if _newest_mtime([cfg.web_dir / s for s in WEB_SOURCES]) > build_id.stat().st_mtime:
        return "web sources changed since the last build"
    return None


def build_web(cfg: LaunchConfig, next_bin: Path, out: TextIO) -> None:
    env = {**os.environ, "POKERMIND_API_URL": cfg.api_url, "NEXT_TELEMETRY_DISABLED": "1"}
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log = cfg.log_dir / "web-build.log"
    with log.open("w") as fh:
        result = subprocess.run(
            [str(next_bin), "build"], cwd=cfg.web_dir, env=env, stdout=fh, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        raise LaunchError(f"the web build failed; see {log}\n{_tail(log)}")
    (cfg.web_dir / ".next" / BUILD_STAMP).write_text(json.dumps({"api_url": cfg.api_url}) + "\n")
    print("  web build ready", file=out)


# -- processes --------------------------------------------------------------------------------


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


@dataclass(slots=True)
class Service:
    name: str
    process: subprocess.Popen[bytes]
    log: Path

    @property
    def alive(self) -> bool:
        return self.process.poll() is None


@dataclass(slots=True)
class Supervisor:
    services: list[Service] = field(default_factory=list)

    def start(
        self, name: str, argv: Sequence[str], cwd: Path, env: dict[str, str], log: Path
    ) -> Service:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("wb")
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # its own process group, children included
            )
        finally:
            fh.close()
        service = Service(name, process, log)
        self.services.append(service)
        return service

    def stop(self, grace: float = STOP_GRACE) -> None:
        """Stop every process group: SIGTERM, wait, then SIGKILL whatever remains."""
        for s in self.services:
            _signal_group(s.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        for s in self.services:
            with contextlib.suppress(subprocess.TimeoutExpired):
                s.process.wait(max(0.0, deadline - time.monotonic()))
        for s in self.services:
            # the group may outlive its leader (e.g. solver workers): always sweep it
            _signal_group(s.process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                s.process.wait(2)
        self.services.clear()


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def wait_http(url: str, service: Service, timeout: float = STARTUP_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not service.alive:
            raise LaunchError(
                f"the {service.name} exited during startup; see {service.log}\n{_tail(service.log)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    raise LaunchError(
        f"the {service.name} did not answer {url} within {timeout:.0f} s; see {service.log}"
    )


# -- the command ------------------------------------------------------------------------------


class _StopRequestedError(Exception):
    pass


def launch(cfg: LaunchConfig, out: TextIO = sys.stdout) -> int:
    print("PokerMind: checking the setup…", file=out)
    check_python()
    summary = check_research(cfg)
    print(f"  {summary}", file=out)
    next_bin = check_node(cfg)
    check_ports(cfg)
    reason = build_reason(cfg)
    if reason is not None:
        print(f"  building the web app ({reason})…", file=out)
        build_web(cfg, next_bin, out)

    supervisor = Supervisor()

    def on_signal(signum: int, _frame: object) -> None:
        raise _StopRequestedError(signal.Signals(signum).name)

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        api_cmd = [
            sys.executable,
            "-m",
            "pokermind.cli",
            "serve",
            "--host",
            cfg.host,
            "--port",
            str(cfg.api_port),
            "--config",
            str(cfg.config),
        ]
        if cfg.hero is not None:
            api_cmd += ["--hero", str(cfg.hero)]
        if cfg.recompute_every is not None:
            api_cmd += ["--recompute-every", str(cfg.recompute_every)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        print("  starting the backend…", file=out)
        api = supervisor.start("backend", api_cmd, cfg.root, env, cfg.log_dir / "api.log")
        wait_http(f"{cfg.api_url}/health", api)
        print("  starting the web app…", file=out)
        web_env = {**env, "NEXT_TELEMETRY_DISABLED": "1"}
        web_cmd = [str(next_bin), "start", "--port", str(cfg.web_port), "--hostname", cfg.host]
        web = supervisor.start("web app", web_cmd, cfg.web_dir, web_env, cfg.log_dir / "web.log")
        wait_http(f"{cfg.web_probe}/", web)
        wait_http(f"{cfg.web_probe}/api/health", web)

        print(f"\n  PokerMind is running:  {cfg.web_url}", file=out)
        print(f"  API docs:              {cfg.api_url}/docs", file=out)
        print(f"  logs:                  {cfg.log_dir}", file=out)
        print("  Press Ctrl-C to stop both services.\n", file=out)
        out.flush()
        if cfg.open_browser and shutil.which("open"):
            subprocess.run(["open", cfg.web_url], check=False)

        while True:
            for s in (api, web):
                if not s.alive:
                    raise LaunchError(
                        f"the {s.name} stopped unexpectedly (exit {s.process.returncode}); "
                        f"see {s.log}\n{_tail(s.log)}"
                    )
            time.sleep(0.5)
    except _StopRequestedError:
        print("\nStopping PokerMind…", file=out)
        return 0
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)  # a second Ctrl-C must not interrupt the cleanup
        supervisor.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        print("  both services stopped.", file=out)


def train_hero(cfg: LaunchConfig, out: TextIO = sys.stdout) -> int:
    """Train the frozen CFR hero (the M4 teacher) at the configuration's path."""
    from pokermind.research import load_research_config

    config = cfg.config if cfg.config.is_absolute() else cfg.root / cfg.config
    research = load_research_config(config)
    target = cfg.root / research.hero.path
    if target.exists():
        print(f"the hero already exists: {target}", file=out)
        return 0
    output_dir = target.parent.parent
    checkpoints = ["1000", "10000", "20000", "50000", str(research.hero.iterations)]
    cmd = [
        sys.executable,
        "-m",
        "pokermind.cli",
        "train-cfr",
        "--game",
        "leduc",
        "--checkpoints",
        *checkpoints,
        "--simulation-hands",
        "100000",
        "--output-dir",
        str(output_dir),
    ]
    print(f"training the CFR hero ({research.hero.iterations:,} iterations, ~30 min)…", file=out)
    out.flush()
    return subprocess.run(cmd, cwd=cfg.root, check=False).returncode


__all__ = [
    "LaunchConfig",
    "LaunchError",
    "Supervisor",
    "build_reason",
    "check_node",
    "check_ports",
    "launch",
    "parse_version",
    "port_free",
    "train_hero",
]
