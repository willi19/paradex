Vendored Wuji Retargeting Subset
================================

Source: https://github.com/wuji-technology/wuji-retargeting
License: MIT, see LICENSE.

This directory vendors the minimal subset Paradex needs for Manus-to-Wuji hand
action retargeting:

- `wuji_retargeting/` core Python retargeter and optimizers
- default Wuji Hand URDF files under `wuji-description/hand/body/urdf`
- Manus retargeting configs under `example/config/retarget_manus_{left,right}.yaml`

Large demo data, visualization assets, MJCF/USD files, and meshes are omitted to
keep the Paradex repository small. Add the omitted assets only when simulation,
visual tuning, or Wuji Hand 2 model support is needed.

Runtime still requires the optimizer dependencies used by the upstream package,
notably `nlopt`, `pinocchio`, `numpy`, `pyyaml`, and `scipy`.
