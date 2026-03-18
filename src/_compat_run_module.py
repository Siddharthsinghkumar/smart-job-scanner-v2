#!/usr/bin/env python3
import runpy
import sys


def run(module_name: str):
    runpy.run_module(module_name, run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: _compat_run_module.py <module>')
    run(sys.argv[1])
