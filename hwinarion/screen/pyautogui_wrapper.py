# pylint: disable=unused-import

# pyautogui.moveTo is too slow, so instead of using the public function, using the lower-level platform module which is
# fast enough.
import platform
import sys

if sys.platform.startswith("java"):
    # from . import _pyautogui_java as pyautogui_module
    raise NotImplementedError("Jython is not yet supported by PyAutoGUI.")
if sys.platform == "darwin":
    from pyautogui import _pyautogui_osx as pyautogui_module  # pylint: disable=unused-import
elif sys.platform == "win32":
    from pyautogui import _pyautogui_win as pyautogui_module  # pylint: disable=unused-import
elif platform.system() == "Linux":
    import Xlib.threaded  # pylint: disable=unused-import
    from pyautogui import _pyautogui_x11 as pyautogui_module  # pylint: disable=unused-import,ungrouped-imports
else:
    raise NotImplementedError(f"Your platform {platform.system()} is not supported by PyAutoGUI.")


# This needs to be imported after Xlib.threaded in Linux, so it's after the imports above
import pyautogui  # pylint: disable=wrong-import-position,wrong-import-order

__ALL__ = ["pyautogui", "pyautogui_module"]
