import subprocess
import json
import os
from paradex.utils.file_io import home_path

ssh_port = 77
repo_path = os.path.join("~", "paradex")


def load_pc_info(pc_list):
    pc_info_path = os.path.join(home_path, "paradex", "config", "environment", "pc.json")
    with open(pc_info_path, 'r') as f:
        pc_info = json.load(f)
    
    if pc_list is None:
        pc_list = list(pc_info.keys())

    for pc_name in pc_list:
        if pc_name not in pc_info:
            raise ValueError(f"PC {pc_name} not found in the configuration file.")
    return pc_info


def git_pull(branch, pc_list=None):
    pc_info = load_pc_info(pc_list)
    for pc_name in pc_list:
        ip = pc_info[pc_name]["ip"]
        remote_cmd = (
            f"cd {repo_path} && "
            f"git fetch origin && "
            f"git reset --hard origin/{branch} --quiet && "
            f"git clean -fd"
        )
        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""
        try:
            print(pc_name)
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")


def run_script(script: str, pc_list = None):    
    pc_info = load_pc_info(pc_list)

    for pc_name in pc_list:
        ip = pc_info[pc_name]["ip"]

        remote_cmd = (
            f"cd {repo_path} && "    
            f"nohup bash -i -c '"
            f"source ~/anaconda3/etc/profile.d/conda.sh && "
            f"conda activate flir_python && "
            f"{script} &' </dev/null > /dev/null 2>&1 & "
        )

        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""

        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")
