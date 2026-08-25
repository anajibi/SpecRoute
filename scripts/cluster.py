#!/usr/bin/env python3
"""Talk to the compute nodes described in cluster.yaml.

Every host/user/port lives in cluster.yaml, so when an IP or username changes you edit
that file and nothing else -- no script has a hostname baked into it.

    python scripts/cluster.py list
    python scripts/cluster.py ping
    python scripts/cluster.py info gpu1
    python scripts/cluster.py run gpu1 'nvidia-smi'
    python scripts/cluster.py setup gpu1              # clone repo + build venv
    python scripts/cluster.py push gpu1 <local-path>  # rsync a file/dir to the node
    python scripts/cluster.py payload gpu1            # push everything in `payload`
    python scripts/cluster.py pull gpu1 <remote-rel-path> [dest]
"""
import argparse
import os
import shlex
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(ROOT, "cluster.yaml")


def load():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def node_by_name(cfg, name):
    for n in cfg["nodes"]:
        if n["name"] == name:
            d = dict(cfg.get("defaults", {}))
            d.update(n)
            return d
    sys.exit(f"no node named {name!r} in cluster.yaml (have: "
             f"{', '.join(n['name'] for n in cfg['nodes'])})")


def ssh_base(n):
    key = os.path.expanduser(n.get("ssh_key", "~/.ssh/id_ed25519"))
    cmd = ["ssh", "-i", key, "-p", str(n.get("port", 22))]
    cmd += list(n.get("ssh_opts", []))
    cmd.append(f"{n['user']}@{n['host']}")
    return cmd


def run_remote(n, command, capture=True, check=False):
    cmd = ssh_base(n) + [command]
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if check and r.returncode != 0:
            sys.exit(f"[{n['name']}] failed: {r.stderr.strip()[:400]}")
        return r.returncode, (r.stdout + r.stderr).rstrip()
    return subprocess.call(cmd), ""


def cmd_list(cfg, args):
    print(f"control: {cfg['control']['user']}@{cfg['control']['hostname']}  "
          f"({cfg['control']['workdir']})")
    print(f"\n{'name':8s} {'target':34s} {'enabled':8s} {'role':10s} gpu")
    for raw in cfg["nodes"]:
        n = {**cfg.get("defaults", {}), **raw}
        hw = n.get("hardware", {})
        tgt = f"{n['user']}@{n['host']}:{n.get('port',22)}"
        print(f"{n['name']:8s} {tgt:34s} {str(n.get('enabled', True)):8s} "
              f"{n.get('role',''):10s} {hw.get('gpu','?')} x{hw.get('gpu_count','?')}")


def cmd_ping(cfg, args):
    ok = True
    for raw in cfg["nodes"]:
        n = {**cfg.get("defaults", {}), **raw}
        if not n.get("enabled", True):
            print(f"{n['name']:8s} SKIPPED (enabled: false)"); continue
        rc, out = run_remote(n, "hostname && uptime -p")
        first = out.splitlines()[0] if out else ""
        print(f"{n['name']:8s} {'OK  ' if rc == 0 else 'FAIL'}  {first}")
        ok &= rc == 0
    sys.exit(0 if ok else 1)


PROBE = (
    'echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd\\; -)"; '
    'echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"; '
    'echo "gpu_free_mib: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"; '
    'echo "gpu_procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"; '
    'echo "cpus: $(nproc)"; '
    'echo "ram_gb: $(free -g | awk \'/^Mem:/{print $2}\')"; '
    'echo "disk_free: $(df -h ~ | tail -1 | awk \'{print $4}\')"; '
    'echo "repo: $(test -d {workdir}/.git && git -C {workdir} log --oneline -1 || echo ABSENT)"; '
    'echo "venv: $(test -x {venv}/bin/python && {venv}/bin/python -c \'import torch;print(torch.__version__, torch.cuda.is_available())\' 2>/dev/null || echo ABSENT)"'
)


def cmd_info(cfg, args):
    n = node_by_name(cfg, args.node)
    probe = PROBE.replace("{workdir}", n["workdir"]).replace("{venv}", n["venv"])
    rc, out = run_remote(n, probe)
    print(f"--- {n['name']}  ({n['user']}@{n['host']}) ---")
    print(out)
    if rc != 0:
        sys.exit(rc)


def cmd_run(cfg, args):
    n = node_by_name(cfg, args.node)
    command = " ".join(args.command)
    rc, _ = run_remote(n, f"cd {shlex.quote(n['workdir'])} 2>/dev/null; {command}", capture=False)
    sys.exit(rc)


def cmd_setup(cfg, args):
    n = node_by_name(cfg, args.node)
    repo, branch = cfg["repo"]["url"], cfg["repo"]["branch"]
    wd = n["workdir"]
    steps = [
        f"test -d {wd}/.git || git clone {repo} {wd}",
        f"cd {wd} && git fetch origin && git checkout {branch} && git pull --ff-only",
    ]
    for s in cfg.get("bootstrap", []):
        steps.append(f"cd {wd} && export PATH=$HOME/.local/bin:$PATH && "
                     + s.replace("{venv}", n["venv"]))
    for i, s in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {s}", flush=True)
        rc, out = run_remote(n, s)
        print(out[-1500:] if out else "(no output)")
        if rc != 0:
            sys.exit(f"step {i} failed on {n['name']} (rc={rc})")
    print("\n--- gotchas ---")
    for g in cfg.get("gotchas", []):
        print(" *", " ".join(g.split()))
    print(f"\nsetup complete on {n['name']}. Next: "
          f"`python scripts/cluster.py payload {n['name']}` to send the data.")


def rsync_to(n, local, remote):
    key = os.path.expanduser(n.get("ssh_key", "~/.ssh/id_ed25519"))
    ssh = f"ssh -i {key} -p {n.get('port',22)} " + " ".join(n.get("ssh_opts", []))
    dest = f"{n['user']}@{n['host']}:{remote}"
    run_remote(n, f"mkdir -p {shlex.quote(os.path.dirname(remote))}")
    cmd = ["rsync", "-avh", "--partial", "--progress", "-e", ssh, local, dest]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_push(cfg, args):
    n = node_by_name(cfg, args.node)
    local = os.path.abspath(args.path)
    rel = os.path.relpath(local, ROOT)
    if rel.startswith(".."):
        sys.exit(f"{local} is outside the repo; give a path under {ROOT}")
    sys.exit(rsync_to(n, local, os.path.join(n["workdir"], rel)))


def cmd_payload(cfg, args):
    n = node_by_name(cfg, args.node)
    total = 0.0
    for item in cfg.get("payload", []):
        if not item.get("required", True) and not args.all:
            print(f"skip (optional): {item['path']}  -- use --all to include"); continue
        local = os.path.join(ROOT, item["path"])
        if not os.path.exists(local):
            print(f"MISSING locally, skipping: {item['path']}"); continue
        print(f"\n=== {item['path']}  (~{item.get('size_gb','?')} GB) ===")
        if rsync_to(n, local, os.path.join(n["workdir"], item["path"])) != 0:
            sys.exit(f"transfer failed: {item['path']}")
        total += float(item.get("size_gb", 0))
    print(f"\npayload done (~{total:.1f} GB)")


def cmd_pull(cfg, args):
    n = node_by_name(cfg, args.node)
    key = os.path.expanduser(n.get("ssh_key", "~/.ssh/id_ed25519"))
    ssh = f"ssh -i {key} -p {n.get('port',22)} " + " ".join(n.get("ssh_opts", []))
    src = f"{n['user']}@{n['host']}:{os.path.join(n['workdir'], args.path)}"
    dest = args.dest or os.path.join(ROOT, args.path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    sys.exit(subprocess.call(["rsync", "-avh", "--partial", "--progress", "-e", ssh, src, dest]))


def main():
    cfg = load()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("ping").set_defaults(fn=cmd_ping)
    q = sub.add_parser("info"); q.add_argument("node"); q.set_defaults(fn=cmd_info)
    q = sub.add_parser("run"); q.add_argument("node"); q.add_argument("command", nargs=argparse.REMAINDER)
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("setup"); q.add_argument("node"); q.set_defaults(fn=cmd_setup)
    q = sub.add_parser("push"); q.add_argument("node"); q.add_argument("path"); q.set_defaults(fn=cmd_push)
    q = sub.add_parser("payload"); q.add_argument("node")
    q.add_argument("--all", action="store_true", help="include optional items (e.g. checkpoints)")
    q.set_defaults(fn=cmd_payload)
    q = sub.add_parser("pull"); q.add_argument("node"); q.add_argument("path")
    q.add_argument("dest", nargs="?"); q.set_defaults(fn=cmd_pull)
    args = p.parse_args()
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
