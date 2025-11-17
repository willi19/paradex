import subprocess
import json
import os
from paradex.utils.path import home_path
from paradex.utils.system import get_pc_list, get_pc_ip

ssh_port = 77
repo_path = os.path.join("~", "paradex")
cache_path = os.path.join("~", "shared_data", "cache")
paradex_cache_path = os.path.join("~", "paradex","paradex","cache")

def load_cache(pc_list=None):
    if pc_list is None:
        pc_list = get_pc_list()
        
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        
        remote_cmd = (
            f"cd {repo_path} && "
            f"cp -r {cache_path} {paradex_cache_path}"
        )
        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""
        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")

def git_pull(branch, pc_list=None):
    if pc_list is None:
        pc_list = get_pc_list()
        
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)
        remote_cmd = (
            f"cd {repo_path} && "
            f"git fetch origin && "
            f"git reset --hard origin/{branch} --quiet && "
            f"git clean -fd"
        )
        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""
        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")


def run_script(script: str, pc_list = None, log=False):    
    if pc_list is None:
        pc_list = get_pc_list()
        
    for pc_name in pc_list:
        ip = get_pc_ip(pc_name)

        # activate conda environment, save script log to null

        if log:
            logoutput = 'test.log'
        else:
            logoutput = '/dev/null'
            
        remote_cmd = (
            f"cd {repo_path} && "    
            f"nohup bash -i -c '"
            f"source ~/anaconda3/etc/profile.d/conda.sh && "
            f"conda activate flir_python && "
            f"{script} &' </dev/null > {logoutput} 2>&1 & "
        )

        ssh_cmd = f"ssh -p {ssh_port} {pc_name}@{ip} \"{remote_cmd}\""

        try:
            subprocess.run(ssh_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{pc_name}] Failed: {e}")
