from isaacgym import gymapi, gymutil

# Initialize Gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Isaac Gym Headless Test")

# Sim params
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.use_gpu_pipeline = True

# Use headless mode (no viewer)
graphics_device_id = 0
compute_device_id = 0

# sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SimType.SIM_CUDA, sim_params)
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create sim")
    exit(1)

# Create env
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

print("Isaac Gym headless simulation initialized.")

# Run simulation for 100 frames
for i in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    print(f"Simulated frame {i+1}")

print("Done.")