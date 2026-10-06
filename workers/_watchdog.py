"""父进程退出时自动结束本进程，避免模型进程残留占着显存。"""

from __future__ import annotations

import os
import sys
import threading
import time


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch_parent(pid: int | None, interval: float = 3.0) -> None:
    if not pid:
        return

    def loop():
        while _alive(pid):
            time.sleep(interval)
        print(">> 主程序已退出，模型进程随之退出", flush=True)
        os._exit(0)

    threading.Thread(target=loop, daemon=True).start()
