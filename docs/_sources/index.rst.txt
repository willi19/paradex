Welcome to paradex's documentation!
===================================

**Paradex** is a comprehensive robotic data collection and control system for manipulation research.

.. note::
   This documentation is under active development.

Quick Start
-----------

Install paradex:

.. code-block:: bash

   pip install -e .

Basic usage example:

.. code-block:: python

   from paradex.io.robot_controller import XArmController
   
   # Initialize robot
   robot = XArmController()
   robot.home_robot(home_pose)
   
   # Start recording
   robot.start(save_path="./data/demo")
   
   # Control loop
   for step in range(100):
       robot.set_action(target_pose)
   
   # Save and cleanup
   robot.end()
   robot.quit()

Features
--------

🤖 **Robot Control**
   Multi-threaded controllers for various robot arms and hands
   (XArm, Franka, Allegro, Inspire)

📹 **Multi-Camera Capture**
   Hardware-synchronized multi-camera recording system

🎮 **Teleoperation**
   VR and motion capture integration (Oculus, Xsens)

📊 **Data Collection**
   Automatic synchronized recording of robot states and visual data

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   getting_started
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules/io
   modules/image
   modules/camera

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`