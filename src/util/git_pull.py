from paradex.io.capture_pc.connect import git_pull, run_script
from paradex.utils.env import get_pcinfo

pc_info = get_pcinfo()
pc_list = list(pc_info.keys())
git_pull("main", pc_list)
