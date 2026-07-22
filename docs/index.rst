Paradex Documentation
=====================

Paradex is a distributed data pipeline for multi-camera robot experiments:
camera services, synchronization, calibration, session recording, processing,
and validation.

The **Guide** starts with the system vocabulary and machine roles, then follows
the normal working order: camera services, calibration, robot control, session
recording, dataset acquisition, image utilities, and post-processing. The
**API Reference** is generated automatically from the ``paradex`` package source.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   overview
   camera_system
   calibration
   robot
   capture
   dataset_acquisition
   image
   process

.. toctree::
   :maxdepth: 2
   :caption: Subsystem API

   camera_system_api
   calibration_api
   robot_api
   capture_api
   dataset_acquisition_api
   image_api
   process_api

.. toctree::
   :maxdepth: 2
   :caption: API Reference

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
