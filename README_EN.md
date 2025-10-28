# AISEED Python Project Template

[한국어](./README.md) | [English](./README_EN.md)

## Introduction

AISEED, an AI company, has many Python projects.  
Unlike before, we need maintenance since we're building actual products beyond just research.

This means we need to focus on **code management** as well as **code writing**.  
However, project maintenance is not easy.

- Code convention unification
- Project structure design
- Test code writing
- Virtual environment and dependency management
- ...

It's inefficient to explain and configure these elements every time when starting a new project or when new personnel join a project.

Therefore, we've created a consistent template for company-wide use.

## Components

The basic components used in this template are as follows:

- [rye](https://rye.astral.sh/guide/) - Project management
  - [uv](https://github.com/astral-sh/uv) - Dependencies
  - [ruff](https://docs.astral.sh/ruff/) - Code rules and formatting
- [mypy](https://mypy.readthedocs.io/en/stable/) - Type support
- [pytest](https://docs.pytest.org/) - Test code
- [pre-commit](https://pre-commit.com/) - Perform pre-tasks during git commit operations

## Getting Started

### Development Environment Setup

- Please install `rye`. ([Installation Guide](https://rye.astral.sh/guide/installation/))

### Project Setup

- Create a repository using this template on Github and clone it.
  ![Github Repository's Use this template](./assets/use-this-template.jpeg)
- Modify the `name`, `version`, `description`, and `authors` in the `pyproject.toml` file in the project folder according to your project.
- Run the following script in the project root directory:
  ```bash
  $ rye sync
  $ pre-commit install
  # If a pre-commit installation error occurs, it's likely that the Python virtual environment is not activated.
  # Try restarting the terminal.
  ```

### Environment Variables Setup
Create an environment variable file `.env` by referring to `.env.sample`.

### Running the Main File
The main file that serves as the program's entry point is `src/main.py`.  
There are 2 scripts to run the main file:

1. `rye run dev`

    Used in development environment.  
    Activates 'development mode' and outputs various warnings such as memory leaks, which is useful.

2. `rye run prod`

    A script with 'development mode' disabled that could cause performance degradation in production environment.  
    Please use this script when actually deploying the program.

Both commands output `Hello, {company_name}` according to the `company_name` environment variable specified in `.env`.

## Project Guidelines

### Dependency Management

Dependencies are managed with `uv` built into `rye` instead of `pip`.  
`uv`, `ruff`, and `rye` are all tools made by the same team, so most commands are compatible.

> [!IMPORTANT]  
> Please distinguish between packages needed for development and packages needed for production.

```bash
# install production dependency
$ rye add numpy

# uninstall production dependency
$ rye remove numpy

# install development dependency
$ rye add --dev pytest

# uninstall development dependency
$ rye remove --dev pytest
```

### Type Check

Find points where type errors occur with `mypy`.

```bash
$ rye run type
```

### Lint

Find points with code convention issues using `ruff`.

```bash
$ rye lint
```

### Running Tests

Run tests in the `tests/` folder with `pytest`.

```bash
# run test
$ rye run test
```

**Writing test code** is a very difficult and extensive topic, so we don't cover how to write test code yet.  
Instead, please write mainly **code usage** so that other members can easily understand the code.

### Git

When `commit`ting work history, use `pre-commit` to inspect changed code.  
Before committing, check for issues with code conventions, typing, and tests using `ruff`, `mypy`, and `pytest`.

## Miscellaneous

### Check Project Environment

```bash
$ rye show
```

### Check List of Executable Scripts

```bash
$ rye run
```

### Script Management

You can add or modify desired scripts in the `[tool.rye.scripts]` section of `pyproject.toml`.

### Changing Python Version

1. Modify to the desired version in `.python-version`

   (Modify `requires-version` in `pyproject.toml` for the target version)

2. Run sync script

   ```bash
   $ rye sync
   ```

### PyTorch Installation
General Python packages are hosted on **PyPI**. On the other hand, `pytorch` has its own separate index.  
Moreover, since there are multiple builds for the same package such as CPU-only builds and CUDA version-specific builds, you need to specify which build to install as a package when installing `pytorch` with `uv`.

The following is an example of using CUDA 12.6 build for Linux and Windows environments, and CPU build for macOS environment.  
If you want to use a different version of CUDA, just change the numbers. (e.g., cu126 -> cu128)

```toml
# pyproject.toml

[tool.uv.sources]
torch = [
  { index = "pytorch-cu126", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]
torchvision = [
  { index = "pytorch-cu126", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

After configuration, install the packages and the installation will be completed with the desired build.

```bash
rye add torch torchvision
```
