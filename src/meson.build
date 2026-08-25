add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

fortranobject_c = run_command(py,
  ['-c', 'import numpy.f2py; import os; print(os.path.join(numpy.f2py.get_include(), "fortranobject.c"))'],
  check: true
).stdout().strip()

py_pkg_name = 'smopt'
py_mod_name = '_smopt'

# Listed explicitly rather than globbed so that Meson can track them as
# real build inputs and reconfigure when one of them changes.
fortran_sources = [
  'smblas.f',
  'smman.f',
  'smprox.f',
  'smslpg.f',
  'smpencf.f',
]

pyf_sources = [
  py_mod_name + '.pyf'
]

message('Fortran sources: ', fortran_sources + pyf_sources)

# Driven through bin/run_f2py.py rather than f2py directly: f2py splits
# its own command line on whitespace, so passing a build directory that
# contains a space fails outright.
f2py_target = custom_target(
  py_mod_name + '_f2py',
  input: pyf_sources,
  output: [py_mod_name + 'module.c', py_mod_name + '-f2pywrappers.f'],
  command: [py, f2py_runner, '@INPUT@', meson.current_build_dir(), '--lower']
)

# Windows wheels must carry the Fortran runtime, since a machine with
# Python but no MinGW has no libgfortran to load.  The -static-* flags
# are understood by the gfortran driver but not by gcc, and Meson would
# otherwise link this mixed C/Fortran target with gcc, so the link
# language is pinned below.  -static-libquadmath in particular only
# exists from GCC 13 on, hence the capability probe rather than a
# hardcoded list.
fc = meson.get_compiler('fortran')

if host_machine.system() == 'windows'
  static_runtime = ['-Wl,-Bstatic', '-lwinpthread', '-Wl,-Bdynamic']
  if not fc.has_link_argument('-static-libquadmath')
    # libgfortran pulls in libquadmath, so without this the wheel would
    # still ask for libquadmath-0.dll at import time.
    static_runtime = ['-Wl,-Bstatic', '-lquadmath', '-lwinpthread',
                      '-Wl,-Bdynamic']
  endif

  fortran_link_args = fc.get_supported_link_arguments([
    '-static-libgfortran',
    '-static-libgcc',
    '-static-libquadmath',
  ]) + static_runtime
else
  fortran_link_args = []
endif

py.extension_module(
  py_mod_name,
  fortran_sources + [f2py_target, fortranobject_c],
  c_args: ['-I' + incdir_numpy, '-I' + incdir_f2py],
  link_args: fortran_link_args,
  link_language: 'fortran',
  dependencies: py_dep,
  subdir: py_pkg_name,
  install: true
)

py.install_sources(
  [
    'smopt/__init__.py',
    'smopt/_bridge.py',
  ],
  subdir: py_pkg_name
)

py.install_sources(
  [
    'smopt/manifold/__init__.py',
    'smopt/manifold/stiefel.py',
  ],
  subdir: py_pkg_name / 'manifold'
)

py.install_sources(
  [
    'smopt/solver/__init__.py',
    'smopt/solver/pencf.py',
    'smopt/solver/slpg.py',
  ],
  subdir: py_pkg_name / 'solver'
)

py.install_sources(
  [
    'smopt/utility/__init__.py',
    'smopt/utility/utility.py',
  ],
  subdir: py_pkg_name / 'utility'
)
