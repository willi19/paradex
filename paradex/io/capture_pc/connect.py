import subprocess
import json
import os
from paradex.utils.file_io import home_path

def reset_and_run(script_name: str, branch: str = "merging"):
    repo_path = "~/paradex"
    ssh_port = 77

    pc_info_path = os.path.join(home_path, "paradex", "config", "environment", "pc.json")
    with open(pc_info_path, 'r') as f:
        pc_info = json.load(f)

    for pc_name, pc in pc_info.items():
        ip = pc["ip"]
        print(f"[{pc_name}] Connecting to {pc_name}@{ip}...")

        remote_cmd = f"""
            cd {repo_path} && \
            git fetch origin && \
            git reset --hard origin/{branch} && \
            git clean -fd && \
            source ~/.bashrc && conda activate flir_python && \
            nohup python {script_name} > logs/{pc_name}.log 2>&1 &
        """

        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} '{remote_cmd}'"

        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")
