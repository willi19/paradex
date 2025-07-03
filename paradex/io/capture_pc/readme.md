# connect.py
### Remote Git & Script Utilities

A small helper module for **synchronized git pulls** and **remote script execution**
across multiple PCs via SSH.

---

#### ✨ Overview

| Function | Purpose |
|----------|---------|
| `load_pc_info(pc_list)` | Load host information from `pc.json` and validate user‑supplied PC names. |
| `git_pull(branch, pc_list=None)` | Force‑update the repository on each remote PC to `origin/<branch>` (`git fetch + reset --hard`). |
| `run_script(script, pc_list=None)` | Execute an arbitrary shell command (e.g. Python script) inside a conda environment (`flir_python`) on each PC using `nohup`. |

The utilities rely on **password‑less SSH** access (e.g. via public‑key auth) to each target host.

---

## 📂 Directory / File Layout

```
~/
└── paradex/                     # remote repo directory on every PC
    └── config/
        └── environment/
            └── pc.json         # host definitions (see below)
```

### `pc.json` example
```json5
{
  "capture01": { "ip": "10.0.0.11" },
  "capture02": { "ip": "10.0.0.12" },
  "analysis":  { "ip": "10.0.0.20" }
}
```

---

## 🛠 Configuration Constants

| Variable | Default | Description |
|----------|---------|-------------|
| `ssh_port` | `77` | SSH port used for all connections. |
| `repo_path` | `~/paradex` | Path to the git repository on every remote PC. |

Adjust these values at the top of the script if your infrastructure is different.

---

## 🕹 Usage

```python
from utils.remote_tools import git_pull, run_script

# 1) Pull the "main" branch on every PC listed in pc.json
git_pull("main")

# 2) Pull the "dev" branch on two specific hosts
git_pull("dev", pc_list=["capture01", "analysis"])

# 3) Run a python script remotely (asynchronous)
cmd = "python src/util/compress_video.py --folder /data/videos"
run_script(cmd, pc_list=["capture01", "capture02"])
```

Each `run_script` invocation:

1. `ssh -p <ssh_port> <pc_name>@<ip>`
2. `cd ~/paradex`
3. `source ~/anaconda3/etc/profile.d/conda.sh && conda activate flir_python`
4. Run the user command under **`nohup`** so it keeps running after the SSH session closes.
5. All stdout/stderr are redirected to `/dev/null`.

---

## ⚠️ Notes & Caveats

1. The repository is forcibly reset with `git reset --hard origin/<branch>`; **all uncommitted changes on the remote will be lost**.
2. Make sure the conda environment name (`flir_python`) exists on every PC, or change it in the script.
3. `run_script()` does not capture output. Use your own logging redirection if you need logs.
4. The local user is assumed to have the same username as `<pc_name>` when connecting via SSH.

---

## 🏷 License

MIT – free to use and modify.
