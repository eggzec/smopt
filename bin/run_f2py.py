#!/usr/bin/env python3
"""Generate the f2py wrappers for a signature file.

f2py re-splits its own command line on whitespace, so a build directory
containing a space -- ``C:\\Users\\First Last\\...``, which is the norm on
Windows -- is torn into fragments and the run fails with a confusing
mixture of "Skipping file" and "Access is denied" errors.

The workaround is to make sure no argument f2py sees ever contains a
separator or a space: the signature file is copied into the build
directory and f2py is invoked from there on the bare file name.

Usage:
    run_f2py.py <signature.pyf> <build-dir> [extra f2py arguments...]
"""

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Copy the signature file into the build directory and run f2py.

    Returns:
        The exit status of the f2py invocation.
    """
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    signature = Path(sys.argv[1]).resolve()
    build_dir = Path(sys.argv[2]).resolve()
    extra_args = sys.argv[3:]

    build_dir.mkdir(parents=True, exist_ok=True)
    staged = build_dir / signature.name
    if staged != signature:
        shutil.copyfile(signature, staged)

    return subprocess.call(  # noqa: S603
        [
            sys.executable,
            "-m",
            "numpy.f2py",
            signature.name,
            "--build-dir",
            ".",
            *extra_args,
        ],
        cwd=str(build_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
