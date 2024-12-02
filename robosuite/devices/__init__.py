from .device import Device
from .keyboard import Keyboard

try:
    from .spacemouse_og import SpaceMouse
    from .spacemouse_hybrid import SpaceMouse as SpaceMouseHybrid
    from .spacemouse_hybrid2 import SpaceMouse as SpaceMouseHybrid2
except ImportError:
    print(
        """Unable to load module hid, required to interface with SpaceMouse.\n
           Only macOS is officially supported. Install the additional\n
           requirements with `pip install -r requirements-extra.txt`"""
    )
