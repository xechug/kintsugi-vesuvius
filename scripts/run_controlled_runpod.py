"""Run one temporary, budget-guarded experiment; download artifacts, then clean up.

Only resource IDs returned by this invocation may be removed. No GitHub writes.
Prepare inputs locally with controlled_ink_benchmark.py before launching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def cli(*args, timeout=90):
    result = subprocess.run(["runpodctl", *args], text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def save(path, data):
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def watchdog(path):
    """Independent stop guard that survives the parent process exiting."""
    while True:
        state = json.loads(path.read_text())
        if state.get("closed"):
            return
        try:
            account = cli("user", timeout=25)
            spent = state["initial_balance"] - account["clientBalance"]
            over = spent >= state["max_spend"] - 0.25 or time.time() >= state["deadline"]
            # Parent can fail between volume creation and pod creation.
            parent_alive = True
            try:
                os.kill(state["parent_pid"], 0)
            except ProcessLookupError:
                parent_alive = False
            if over or not parent_alive:
                if state.get("pod_id"):
                    cli("pod", "delete", state["pod_id"], timeout=30)
                state["guard_stopped_pod"] = True
                state["guard_reason"] = "budget/time limit" if over else "parent exited"
                save(path, state)
                return
        except Exception as exc:
            print("Guard read failed:", str(exc), flush=True)
            if time.time() >= state["deadline"] and state.get("pod_id"):
                try:
                    cli("pod", "delete", state["pod_id"], timeout=30)
                    return
                except Exception:
                    pass
        time.sleep(30)


def run(args):
    repo = Path(__file__).resolve().parents[1]
    reader = args.study == "reader-v3"
    hybrid = args.study == "reader-kintsugi"
    if hybrid:
        study = "reader_kintsugi_20261025"
        resource_name = "reader-kintsugi-20261025"
    elif reader:
        study = "reader_v3_survey_20261025"
        resource_name = "reader-v3-survey-20261025"
    else:
        study = "controlled_ink_20261025"
        resource_name = "kintsugi-control-20261025"
    out = repo / "runs" / study
    data = repo / "data" / study
    if not (data / "hashes.json").exists() or not (repo / "scripts/official_ink_reference.py").is_file():
        raise RuntimeError("Prepare and validate local inputs before renting a GPU")
    if (out / "frozen_checkpoints.json").exists() or (out / "artifacts.json").exists():
        raise RuntimeError("This output already contains an experiment; do not overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "runpod_state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    active_pods = cli("pod", "list")
    existing_volumes = cli("network-volume", "list")
    if previous and not previous.get("closed"):
        if previous.get("guard_stopped_pod") and not active_pods:
            previous["closed"] = True
            save(state_path, previous)
        else:
            raise RuntimeError("An existing attempt must be closed before starting another")
    account = cli("user")
    initial_balance = previous["initial_balance"] if previous else account["clientBalance"]
    remaining = args.max_spend - (initial_balance - account["clientBalance"])
    if remaining < 1.0 or account["currentSpendPerHr"] > 0.02 or active_pods:
        raise RuntimeError("Balance or existing account spend does not match the approved budget")
    gpu = next(g for g in cli("gpu", "list") if g["gpuId"] == args.gpu_id)
    price = gpu["securePricePerHr"]
    stock = next((x for x in gpu.get("dataCenterAvailability", []) if x["dataCenterId"] == args.data_center), None)
    if price is None or price > 0.90 or not stock or stock["stockStatus"].lower() in ("none", "unavailable"):
        raise RuntimeError("Approved GPU price/stock is unavailable; no resources created")
    template = cli("template", "get", "runpod-torch-v280")
    keys = cli("ssh", "list-keys")
    if not keys.get("keys"):
        raise RuntimeError("No existing account SSH key")
    if previous:
        save(out / f"runpod_attempt_{previous.get('attempt',1)}.json", previous)
    state = {"parent_pid": os.getpid(), "initial_balance": initial_balance,
             "max_spend": args.max_spend, "gpu_price_per_hour": price,
             "image": template.get("imageName"), "stock": stock,
             "attempt": previous.get("attempt",1)+1 if previous else 1,
             "started_at": time.time(), "deadline": time.time() + min(5*3600,(remaining-0.25)/(price+0.01)*3600),
             "closed": False}
    save(state_path, state)
    log = (out / "budget_guard.log").open("a")
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(state_path)],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    backed_up = False
    worker_started = False
    try:
        prev_vol_id = previous.get("volume_id") if previous else None
        reused = next((v for v in existing_volumes if v["id"] == prev_vol_id), None) if prev_vol_id else None
        if reused:
            volume = reused
            print("REUSING_VOLUME", volume["id"], flush=True)
        else:
            volume = cli("network-volume", "create", "--name", resource_name, "--size", "10", "--data-center-id", args.data_center)
        state["volume_id"] = volume["id"]
        save(state_path, state)
        state["volume_readback"] = cli("network-volume", "get", volume["id"])
        pod = cli("pod", "create", "--name", resource_name, "--template-id", "runpod-torch-v280",
                  "--gpu-id", gpu["gpuId"], "--cloud-type", "SECURE", "--data-center-ids", args.data_center,
                  "--network-volume-id", volume["id"], "--container-disk-in-gb", "30", "--volume-in-gb", "0",
                  "--volume-mount-path", "/workspace", "--ports", "22/tcp", "--min-cuda-version", "12.8", "--ssh")
        state["pod_id"] = pod["id"]
        save(state_path, state)
        readback = cli("pod", "get", pod["id"])
        state["pod_readback"] = {k: readback.get(k) for k in ["id", "desiredStatus", "imageName", "costPerHr", "networkVolumeId", "ports"]}
        save(state_path, state)
        print("CREATED", pod["id"], "GPU_USD_PER_HOUR", price, flush=True)
        ready_deadline = time.time()+600
        info = None
        while time.time() < ready_deadline:
            try:
                info = cli("ssh", "info", pod["id"])
                if info.get("ip") and info.get("port"):
                    break
                candidate = cli("pod", "get", pod["id"]).get("ssh", {})
                if candidate.get("ip") and candidate.get("port"):
                    try:
                        with socket.create_connection((candidate["ip"], int(candidate["port"])), timeout=3) as connection:
                            connection.settimeout(3)
                            if connection.recv(256).startswith(b"SSH-"):
                                info = candidate
                                break
                    except OSError:
                        pass
            except RuntimeError:
                pass
            time.sleep(10)
        if not info or not info.get("ip") or not info.get("port"):
            raise RuntimeError("Pod did not expose direct SSH within ten minutes")
        key = info["ssh_key"]["path"]
        ssh = ["ssh", "-i", key, "-p", str(info["port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile="+str(out/"known_hosts"), "root@"+info["ip"]]
        scp = ["scp", "-i", key, "-P", str(info["port"]), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile="+str(out/"known_hosts")]

        def remote(code, timeout=120):
            return subprocess.run(ssh + ["python3 -c " + shlex.quote(code)], text=True, capture_output=True, timeout=timeout, check=True).stdout

        for attempt in range(20):
            try:
                print(remote("import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name())"), flush=True)
                break
            except (subprocess.SubprocessError, OSError):
                if attempt == 19:
                    raise
                time.sleep(5)
        has_data = remote(f"import pathlib; print((pathlib.Path('/workspace/kintsugi/data/{study}/hashes.json')).is_file())").strip() == "True"
        descriptor, filename = tempfile.mkstemp(prefix="kintsugi-control-", suffix=".tar.gz")
        os.close(descriptor)
        bundle = Path(filename)
        try:
            with tarfile.open(bundle, "w:gz", compresslevel=1) as tar:
                if hybrid:
                    names = ["vesuvius_code", "scripts/reader_v3_survey.py", "scripts/train_reader_kintsugi.py",
                             f"configs/{study}.json"] + ([] if has_data else [f"data/{study}"])
                elif reader:
                    names = ["vesuvius_code", "scripts/reader_v3_survey.py",
                             f"configs/{study}.json"] + ([] if has_data else [f"data/{study}"])
                else:
                    names = ["vesuvius_code", "scripts/controlled_ink_benchmark.py", "scripts/official_ink_reference.py",
                             f"configs/{study}.json"] + ([] if has_data else [f"data/{study}"])
                for name in names:
                    tar.add(repo/name, arcname=name, filter=lambda member: None if "__pycache__" in member.name else member)
            subprocess.run(scp + [str(bundle), f"root@{info['ip']}:/workspace/bundle.tar.gz"], check=True, timeout=600)
        finally:
            bundle.unlink(missing_ok=True)
        print(remote("import pathlib,tarfile; p=pathlib.Path('/workspace/kintsugi'); p.mkdir(exist_ok=True); tarfile.open('/workspace/bundle.tar.gz').extractall(p,filter='data')"), flush=True)
        print(remote("import subprocess,sys; subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','--no-deps','numcodecs==0.16.5','pillow','einops==0.8.2','timm==1.0.25'],check=True)", timeout=300), flush=True)
        print(remote("import subprocess,sys; subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','huggingface_hub==0.36.0','PyYAML'],check=True)", timeout=180), flush=True)
        if reader or hybrid:
            print(remote("import subprocess,sys; subprocess.run([sys.executable,'-m','pip','install','--break-system-packages','zarr==3.2.1','tifffile'],check=True)", timeout=180), flush=True)
        if hybrid:
            commands = [('reader_kintsugi', ['scripts/train_reader_kintsugi.py'])]
        elif reader:
            commands = [('reader_v3_survey', ['scripts/reader_v3_survey.py', '--stage', 'run'])]
        else:
            commands = [('controlled_training', ['scripts/controlled_ink_benchmark.py', '--stage', 'run']),
                        ('official_reference', ['scripts/official_ink_reference.py'])]
        worker = """import json,os,pathlib,subprocess,sys,time,traceback
p=pathlib.Path('/workspace/kintsugi'); os.chdir(p)
o=p/'runs/STUDY_NAME'; o.mkdir(parents=True,exist_ok=True)
status=o/'worker_status.json'
try:
 for phase,arguments in WORKER_COMMANDS:
  status.write_text(json.dumps({'phase':phase,'running':True}))
  subprocess.run([sys.executable,*arguments],check=True,env={**os.environ,'PYTHONPATH':str(p)})
 status.write_text(json.dumps({'phase':'complete','running':False,'success':True}))
except BaseException:
 status.write_text(json.dumps({'phase':'failed','running':False,'success':False,'error':traceback.format_exc()}))
 raise
""".replace("STUDY_NAME", study).replace("WORKER_COMMANDS", repr(commands))
        code = "import pathlib,subprocess,sys; p=pathlib.Path('/workspace/kintsugi'); (p/'worker.py').write_text("+repr(worker)+"); f=(p/'worker.log').open('w'); subprocess.Popen([sys.executable,'-u','worker.py'],cwd=p,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)"
        remote(code)
        worker_started = True
        worker_state = {}
        while time.time() < state["deadline"]:
            live = json.loads(state_path.read_text())
            if live.get("guard_stopped_pod"):
                raise RuntimeError("Budget guard stopped the pod")
            try:
                output = remote("import pathlib,json; p=pathlib.Path('/workspace/kintsugi'); s=p/" + repr("runs/"+study+"/worker_status.json") + "; print(s.read_text() if s.exists() else '{}'); print('\\n'.join((p/'worker.log').read_text().splitlines()[-4:]))")
                lines = output.splitlines()
                worker_state = json.loads(lines[0]) if lines else {}
                print(output, flush=True)
                if worker_state.get("running") is False:
                    break
            except (subprocess.SubprocessError, OSError) as exc:
                print("Transient SSH poll error, retrying:", str(exc), flush=True)
            time.sleep(20)
        state["worker_status"] = worker_state
        remote("import pathlib,shutil; p=pathlib.Path('/workspace/kintsugi'); shutil.copyfile(p/'worker.log',p/" + repr("runs/"+study+"/worker.log") + ")")
        subprocess.run(scp + ["-r", f"root@{info['ip']}:/workspace/kintsugi/runs/{study}/.", str(out)], check=True, timeout=600)
        # Preserve new weights and raw maps before removing either rented resource.
        index_path = out/"frozen_checkpoints.json"
        if index_path.exists():
            for item in json.loads(index_path.read_text()):
                digest = hashlib.sha256((out/item["checkpoint"]).read_bytes()).hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError("Downloaded checkpoint checksum mismatch")
        if (out/"artifacts.json").exists():
            for relative, expected in json.loads((out/"artifacts.json").read_text()).items():
                if hashlib.sha256((out/relative).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Downloaded survey artifact checksum mismatch: "+relative)
        backed_up = True
        state["artifacts_backed_up"] = True
        if not worker_state.get("success"):
            raise RuntimeError("Worker failed; artifacts downloaded for diagnosis: "+str(worker_state.get("error")))
    finally:
        if state.get("pod_id"):
            if any(x["id"] == state["pod_id"] for x in cli("pod", "list")):
                cli("pod", "delete", state["pod_id"])
            if any(x["id"] == state["pod_id"] for x in cli("pod", "list")):
                raise RuntimeError("Pod deletion was not confirmed")
            state["pod_deleted"] = True
        if state.get("volume_id") and (backed_up or not worker_started):
            cli("network-volume", "delete", state["volume_id"])
            if any(x["id"] == state["volume_id"] for x in cli("network-volume", "list")):
                raise RuntimeError("Volume deletion was not confirmed")
            state["volume_deleted"] = True
        final = cli("user")
        state["final_balance"] = final["clientBalance"]
        state["spent"] = state["initial_balance"] - final["clientBalance"]
        state["current_spend_per_hour"] = final["currentSpendPerHr"]
        state["closed"] = True
        save(state_path, state)
        print("FINAL_BUDGET", json.dumps({k:state.get(k) for k in ["spent","final_balance","pod_deleted","volume_deleted"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-spend", type=float, default=5.0)
    parser.add_argument("--study", choices=("controlled", "reader-v3", "reader-kintsugi"), default="controlled")
    parser.add_argument("--data-center", default="EU-RO-1")
    parser.add_argument("--gpu-id", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--watchdog", type=Path)
    options = parser.parse_args()
    if options.watchdog:
        watchdog(options.watchdog)
    else:
        if not 0 < options.max_spend <= 5.0:
            raise ValueError("This experiment is authorized for at most 5 USD")
        run(options)
