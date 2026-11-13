"""Wrapper: read OPENROUTER_API_KEY from the permanent per-user registry env
var (needed because this session's shell process started before the key was
registered, so os.environ doesn't have it) and set it in-process, then run
the target script (given as argv[1]) with its own remaining argv."""
import os
import runpy
import sys
import winreg

with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
    val, _ = winreg.QueryValueEx(key, "OPENROUTER_API_KEY")
os.environ["OPENROUTER_API_KEY"] = val

target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
