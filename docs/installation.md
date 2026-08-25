# Installation

`<Project Name>` can be installed from PyPI or directly from source via GitHub.

---

## [PyPI](https://pypi.org/project/<Project Name>)

For using the PyPI package in your project, add it to your configuration file:

=== "pyproject.toml"

    ```toml
    [project.dependencies]
    <Project Name> = "*" # (1)!
    ```

    1. Specifying a version is recommended

=== "requirements.txt"

    ```
    <Project Name>>=0.1.0
    ```

### pip

=== "Installation for user"

    ```bash
    pip install --upgrade --user <Project Name> # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

=== "Installation in virtual environment"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --require-virtualenv --upgrade <Project Name> # (1)!
    ```

    1. You may need to use `pip3` instead of `pip` depending on your Python installation.

    !!! note
        The command to activate the virtual environment depends on your platform and shell.
        [More info](https://docs.python.org/3/library/venv.html#how-venvs-work)

### uv

=== "Adding to uv project"

    ```bash
    uv add <Project Name>
    uv sync
    ```

=== "Installing to uv environment"

    ```bash
    uv venv
    uv pip install <Project Name>
    ```

### pipenv

```bash
pipenv install <Project Name>
```

### poetry

```bash
poetry add <Project Name>
```

### pdm

```bash
pdm add <Project Name>
```

### hatch

```bash
hatch add <Project Name>
```

---

## [GitHub](https://github.com/eggzec/<Project Name>)

Install the latest development version directly from the repository:

```bash
pip install --upgrade "git+https://github.com/eggzec/<Project Name>.git#egg=<Project Name>"
```

### Building locally

Clone and build from source if you want to modify or test local changes:

```bash
git clone https://github.com/eggzec/<Project Name>.git
cd <Project Name>
pip install -e .
```

---
