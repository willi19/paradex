Welcome to paradex's documentation!
===================================

.. image:: _static/paradex_demo.gif
   :align: center
   :width: 80%

Paradex is a comprehensive robotic data collection and control system for manipulation research. 
It fully supports **Linux** and is distributed under MIT license.

Examples
--------

.. figure:: _static/xarm_demo.png
   :align: center
   :width: 600px
   
   paradex example: XArm manipulation

.. code-block:: python

   from paradex.io.robot_controller import XArmController
   
   robot = XArmController()
   robot.home_robot(home_pose)
   robot.start(save_path="./data/demo")

.. figure:: _static/camera_demo.png
   :align: center
   :width: 600px
   
   paradex example: Multi-camera capture

.. code-block:: python

   from paradex.io.camera.camera_loader import CameraManager
   
   cameras = CameraManager(mode="video", syncMode=True)
   cameras.start(save_dir="./captures")

.. toctree::
   :maxdepth: 1
   :caption: GET STARTED
   
   installation
   quickstart
   license

.. toctree::
   :maxdepth: 2
   :caption: PARADEX CORE
   
   modules/io
   modules/camera
   modules/image
   modules/robot
   modules/utils

.. toctree::
   :maxdepth: 1
   :caption: RELATED SOFTWARE
   
   related/isaac_sim
   related/mujoco
   related/ros

.. toctree::
   :maxdepth: 1
   :caption: 🚀 GET STARTED
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 📚 API REFERENCE
   
   modules/io
   modules/camera

.. toctree::
   :maxdepth: 1
   :caption: 🔗 RELATED SOFTWARE
   
   related/isaac_sim