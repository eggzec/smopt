project(
    'smopt',
    'fortran', 'c',
    version: run_command(['python', 'bin/get_version.py'], check: true).stdout().strip()
)

# Passed down to src/ ; see the script for the f2py quoting bug it works
# around.
f2py_runner = files('bin/run_f2py.py')

subdir('src')
