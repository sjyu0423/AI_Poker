"""M9: the one-command launcher (checks, build staleness, process-group shutdown)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from pokermind.cli import main
from pokermind.launcher import (
    BUILD_STAMP,
    LaunchConfig,
    LaunchError,
    Supervisor,
    build_reason,
    check_node,
    check_ports,
    parse_version,
    port_free,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_version() -> None:
    assert parse_version("v26.12.0\n") == (26, 12, 0)
    assert parse_version("Python 3.12") == (3, 12)
    with pytest.raises(ValueError):
        parse_version("none")


def test_port_conflicts_are_reported_with_a_fix() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not port_free("127.0.0.1", busy)
        with pytest.raises(LaunchError, match=rf"port {busy} \(API\).*--api-port"):
            check_ports(LaunchConfig(ROOT, api_port=busy, web_port=busy + 1))
        with pytest.raises(LaunchError, match=r"--web-port"):
            check_ports(LaunchConfig(ROOT, api_port=busy + 1, web_port=busy))
    with pytest.raises(LaunchError, match="must differ"):
        check_ports(LaunchConfig(ROOT, api_port=9100, web_port=9100))


def _web(tmp_path: Path) -> LaunchConfig:
    (tmp_path / "web" / "src").mkdir(parents=True)
    (tmp_path / "web" / "src" / "page.tsx").write_text("x")
    return LaunchConfig(tmp_path)


def test_build_reason(tmp_path: Path) -> None:
    cfg = _web(tmp_path)
    assert build_reason(cfg) == "no production build yet"
    build = tmp_path / "web" / ".next"
    build.mkdir()
    (build / "BUILD_ID").write_text("id")
    assert "unknown API" in (build_reason(cfg) or "")
    (build / BUILD_STAMP).write_text(json.dumps({"api_url": "http://127.0.0.1:9999"}))
    assert "not http://127.0.0.1:8000" in (build_reason(cfg) or "")
    (build / BUILD_STAMP).write_text(json.dumps({"api_url": cfg.api_url}))
    old = time.time() - 100
    os.utime(tmp_path / "web" / "src" / "page.tsx", (old, old))
    assert build_reason(cfg) is None
    (tmp_path / "web" / "src" / "page.tsx").write_text("changed")
    assert build_reason(cfg) == "web sources changed since the last build"
    forced = LaunchConfig(tmp_path, rebuild=True)
    assert build_reason(forced) == "rebuild requested"


def test_node_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _web(tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(LaunchError, match=r"Node\.js is not installed"):
        check_node(cfg)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node")
    with pytest.raises(LaunchError, match="too old"):
        check_node(cfg, run=lambda *a: "v18.20.0")
    with pytest.raises(LaunchError, match=r"npm ci"):
        check_node(cfg, run=lambda *a: "v22.1.0")
    nxt = tmp_path / "web" / "node_modules" / ".bin" / "next"
    nxt.parent.mkdir(parents=True)
    nxt.write_text("")
    assert check_node(cfg, run=lambda *a: "v22.1.0") == nxt


def test_missing_hero_explains_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(ROOT)
    code = main(["launch", "--hero", str(tmp_path / "missing.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert "not found" in err and "--train-hero" in err


def test_launch_must_run_from_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["launch"]) == 2
    assert "repository root" in capsys.readouterr().err


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_supervisor_stops_the_whole_process_group(tmp_path: Path) -> None:
    sup = Supervisor()
    pidfile = tmp_path / "child.pid"
    # the leader starts a grandchild that ignores SIGTERM: only the SIGKILL sweep stops it
    script = (
        f"import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', "
        f"'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        f"time.sleep(60)\n"
    )
    service = sup.start(
        "test", [sys.executable, "-c", script], tmp_path, dict(os.environ), tmp_path / "log"
    )
    deadline = time.monotonic() + 10
    while not pidfile.exists() or not pidfile.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    grandchild = int(pidfile.read_text())
    assert service.alive and _alive(grandchild)
    sup.stop(grace=1.0)
    assert not service.alive
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)
    assert sup.services == []


@pytest.mark.skipif(
    os.environ.get("POKERMIND_E2E") != "1",
    reason="end-to-end launch (builds the web app); set POKERMIND_E2E=1",
)
def test_launch_end_to_end(tmp_path: Path) -> None:
    """Real launch on spare ports: both services answer, Ctrl-C leaves no process behind."""
    api, web = 18_765, 13_765
    log = tmp_path / "launch.log"
    with log.open("w") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pokermind.cli", "launch", "--api-port", str(api),
             "--web-port", str(web), "--demo"],
            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
        )  # fmt: skip
    try:
        deadline = time.monotonic() + 240
        while "PokerMind is running" not in log.read_text():
            assert proc.poll() is None, log.read_text()
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.5)
        with urllib.request.urlopen(f"http://127.0.0.1:{web}/api/health", timeout=5) as r:
            assert json.load(r)["status"] == "ok"
        with urllib.request.urlopen(f"http://127.0.0.1:{web}/", timeout=5) as r:
            assert b"PokerMind" in r.read()
    finally:
        proc.send_signal(signal.SIGINT)
        assert proc.wait(30) == 0, log.read_text()
    assert "both services stopped" in log.read_text()
    assert port_free("127.0.0.1", api) and port_free("127.0.0.1", web)
    out = subprocess.run(["pgrep", "-f", f"--port {api}"], capture_output=True, text=True)
    assert out.stdout.strip() == ""
