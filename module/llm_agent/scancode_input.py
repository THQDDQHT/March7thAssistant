"""前台输入模式下用硬件扫描码发键（SendInput + KEYEVENTF_SCANCODE）。

M7A 的前台输入用 pyautogui.keyDown/keyUp，底层是 keybd_event(vk, 0, ...)，扫描码是 0。
游戏的战斗快捷键（e 切战技、数字 1–4 放终结技、确认终结技的空格）读的是硬件扫描码，所以收不到。
实测：SendInput 只带扫描码（wVk=0，wScan=MapVirtualKeyW(vk, 0)，flags=KEYEVENTF_SCANCODE）时，e、2、空格全部有效。

只用于前台输入模式（LocalInput）；后台输入、云游戏、离线模式都不走这里。
非 Windows 环境可以导入，导入时不调用 windll；调用发键函数时才取 user32，取不到就返回 False。
支持的键：a–z、0–9、space、enter、esc。返回值表示 SendInput 是否成功（返回的事件数为 1）。
"""
import ctypes
import sys
import time
from ctypes import wintypes
from typing import Callable, Optional

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INPUT_KEYBOARD = 1
MAPVK_VK_TO_VSC = 0
DEFAULT_HOLD = 0.15

_SPECIAL_VK = {"space": 0x20, "enter": 0x0D, "esc": 0x1B}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUnion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUnion)]


def virtual_key(key) -> Optional[int]:
    """键名 → 虚拟键码：a–z 是 ord(大写)，0–9 是 ord(数字)，space 0x20，enter 0x0D，esc 0x1B；不支持返回 None。"""
    if not isinstance(key, str):
        return None
    name = key.strip().lower()
    if name in _SPECIAL_VK:
        return _SPECIAL_VK[name]
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name.upper())
    return None


def _user32():
    if sys.platform != "win32":
        return None
    return ctypes.windll.user32


def scan_code(key, user32=None) -> Optional[int]:
    """键名 → 硬件扫描码（MapVirtualKeyW(vk, 0)）；不支持的键或取不到返回 None。"""
    vk = virtual_key(key)
    user32 = user32 if user32 is not None else _user32()
    if vk is None or user32 is None:
        return None
    scan = int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC))
    return scan or None


def _send(scan: int, up: bool, user32) -> bool:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    event = INPUT(type=INPUT_KEYBOARD, u=_INPUTUnion(ki=KEYBDINPUT(0, scan, flags, 0, 0)))
    return user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) == 1


def key_down(key, user32=None) -> bool:
    user32 = user32 if user32 is not None else _user32()
    scan = scan_code(key, user32)
    if scan is None or user32 is None:
        return False
    return _send(scan, False, user32)


def key_up(key, user32=None) -> bool:
    user32 = user32 if user32 is not None else _user32()
    scan = scan_code(key, user32)
    if scan is None or user32 is None:
        return False
    return _send(scan, True, user32)


def press(key, hold_seconds: float = DEFAULT_HOLD, sleep: Callable = time.sleep, user32=None) -> bool:
    """按下 → 等 hold_seconds → 松开。按下成功后无论如何都会发松开（避免键卡在按下状态）。"""
    user32 = user32 if user32 is not None else _user32()
    scan = scan_code(key, user32)
    if scan is None or user32 is None:
        return False
    if not _send(scan, False, user32):
        return False
    try:
        sleep(hold_seconds)
    finally:
        released = _send(scan, True, user32)
    return released


def use_for(auto) -> bool:
    """这个 auto 是不是前台输入（LocalInput）且在 Windows 上：是才改用扫描码发键。

    只认 LocalInput 的实例：后台输入（BackgroundInput）、云游戏（CdpInput）、离线替身、测试里的假 auto 都不算。
    """
    if sys.platform != "win32":
        return False
    try:
        from module.automation.local_input import LocalInput
    except Exception:
        return False
    return isinstance(getattr(auto, "input_handler", None), LocalInput)
