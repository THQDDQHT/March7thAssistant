"""必须原样向上抛出的上游异常。

上游的后台输入模式会抛出两类需要调用方处理的异常（都是 RuntimeError 的子类）：
- InputInterrupted：后台输入无法继续且需要用户处理（例如窗口持续最小化、无法读写光标）；
- GameWindowLost：游戏进程已退出或长时间找不到游戏窗口。
暂停和停止指令（utils.pause.RunStopRequested）是 BaseException，`except Exception` 本来就接不到。

接管模块里所有 `except Exception` 都要先放行这两类异常，不能当作普通失败吞掉，
否则任务会在窗口丢失、需要用户处理的情况下继续空转，或者在不该重启游戏的地方被重启。
"""

_passthrough = None


def passthrough_exceptions() -> tuple:
    """返回需要原样抛出的异常类型元组（第一次调用时再导入，导入失败说明没有后台输入模块，返回空元组）。"""
    global _passthrough
    if _passthrough is None:
        try:
            from module.automation.background_input import GameWindowLost, InputInterrupted
            _passthrough = (InputInterrupted, GameWindowLost)
        except Exception:
            _passthrough = ()
    return _passthrough


def reraise_passthrough(error: BaseException):
    """在 `except Exception as e:` 的开头调用：如果是必须向上抛的异常就原样抛出。"""
    if isinstance(error, passthrough_exceptions()):
        raise error
