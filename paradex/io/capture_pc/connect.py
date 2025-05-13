import subprocess
import json
import os
from paradex.utils.file_io import home_path

def reset_and_run(script_name: str, pc_list = None, branch: str = "merging"):
    repo_path = "~/paradex"
    ssh_port = 77

    pc_info_path = os.path.join(home_path, "paradex", "config", "environment", "pc.json")
    with open(pc_info_path, 'r') as f:
        pc_info = json.load(f)

    if pc_list is None:
        pc_list = list(pc_info.keys())
        
    for pc_name in pc_list:
        if pc_name not in pc_info:
            raise ValueError(f"PC {pc_name} not found in the configuration file.")
    
    for pc_name in pc_list:
        ip = pc_info[pc_name]["ip"]
        remote_cmd = (
            f"cd {repo_path} && "
            f"git fetch origin && "
            f"git reset --hard origin/{branch} --quiet && "
            f"git clean -fd &&"
            
            f"nohup bash -i -c '"
            f"source ~/anaconda3/etc/profile.d/conda.sh && "
            f"conda activate flir_python && "
            
            f"python {script_name} &' </dev/null > /dev/null 2>&1 & "
        )

        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""

        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")
