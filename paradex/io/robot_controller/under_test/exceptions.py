"""
Exceptions for the Inspire Hand library.
"""

class InspireHandError(Exception):
    """Base exception for all Inspire Hand errors."""
    pass

class ConnectionError(InspireHandError):
    """Exception raised when there's an issue connecting to the hand."""
    pass

class CommandError(InspireHandError):
    """Exception raised when a command fails to execute."""
    pass 