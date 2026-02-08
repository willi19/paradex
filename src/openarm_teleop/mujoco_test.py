import mujoco
import mujoco.viewer

# 1. Load the MuJoCo model and create data
model = mujoco.MjModel.from_xml_path('v1/openarm_bimanual.xml')
data = mujoco.MjData(model)

# 2. Simulation viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 3. Advance physics simulation by one step
        mujoco.mj_step(model, data)

        # 4. Sync viewer (update screen)
        viewer.sync()