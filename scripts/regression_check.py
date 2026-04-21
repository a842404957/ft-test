#!/usr/bin/env python3

import importlib.util
import os
import sys
import unittest
from pathlib import Path


REQUIRED_MODULES = ('torch', 'pandas')


def _find_missing_modules():
    missing = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    missing_modules = _find_missing_modules()
    if missing_modules:
        print('Regression prerequisites are missing for the current interpreter.')
        print(f'python = {sys.executable}')
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print(f'conda_env = {conda_env}')
        print('missing_modules = {}'.format(', '.join(missing_modules)))
        print('Use a Python environment that has these packages installed.')
        print('Example: conda run -n <env> python scripts/regression_check.py')
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(repo_root / 'tests'),
        pattern='test_*.py',
        top_level_dir=str(repo_root),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
