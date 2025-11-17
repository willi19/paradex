from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

rcc = remote_camera_controller("reset_camera_server.py")
rcc.reset()
rcc.end()