"""Reproduce status-file replacement failure without touching run artifacts."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile


def main():
    parent = Path(__file__).resolve().parents[1] / 'result/concurrent_fleet_v8_fresh_train20_20261018'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    with tempfile.TemporaryDirectory(prefix='replace_diagnostic_', dir=parent) as folder:
        target = Path(folder) / 'status.json'
        temp = target.with_suffix('.tmp')
        target.write_text('{}', encoding='utf-8')
        for i in range(1000):
            temp.write_text(json.dumps({'step': i}), encoding='utf-8')
            temp.replace(target)
        print('unlocked replacements: 1000/1000 passed')
        # A reader may allow other readers/writers but deny deletion/rename.
        handle = kernel.CreateFileW(str(target), 0x80000000, 0x1 | 0x2,
                                    None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            temp.write_text('{"step":1000}', encoding='utf-8')
            try:
                os.replace(temp, target)
            except OSError as error:
                print(json.dumps({'locked_replace': type(error).__name__,
                                  'winerror': error.winerror, 'errno': error.errno}))
            else:
                raise AssertionError('Expected reader handle to block replacement')
        finally:
            kernel.CloseHandle(handle)
        temp.replace(target)
        print('replacement after reader closed: passed')


if __name__ == '__main__':
    main()
