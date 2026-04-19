#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(repo_root / 'tests'),
        pattern='test_*.py',
        top_level_dir=str(repo_root),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
