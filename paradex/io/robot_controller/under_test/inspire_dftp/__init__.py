"""
Inspire Hand RH56 Control Library

A Python library for controlling the Inspire Hand robotic hands.
Supports both RH56dfq (legacy) and RH56DFTP models.
"""

from .hand_rh56dftp import InspireHandRH56DFTP, ContactStatus
from .exceptions import InspireHandError, ConnectionError, CommandError

__version__ = '0.1.0'

__all__ = [
    'InspireHandRH56DFTP',
    'ContactStatus',
    'InspireHandError',
    'ConnectionError',
    'CommandError',
] 