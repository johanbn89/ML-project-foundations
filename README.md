# ML-Project-Foundations

This repository demonstrates how to structure a **production-ready, end-to-end machine learning project**.

The focus is not on models and training code, but on the **infrastructure and engineering practices** that make ML projects maintainable, testable, and scalable over time.

## Repository Setup

### 1. Install `uv`

Follow the installation instructions here:  
https://astral.sh/uv

### 2. Clone the repository
```bash
git clone <repo-url>
cd ml-project-foundations
```
### 3. Install Python and dependencies
```bash
uv sync --all-groups
python scripts/setup_local.py
```
setup_local.py install local deps, eg. ml-project-foundations-data-quarry.

Notice that `uv sync`:

- Creates (or reuses) the project virtual environment
- Ensures a compatible Python version is available (based on `.python-version`), installing it if necessary
- Installs all dependencies from `pyproject.toml` / `uv.lock`
- Installs the current project in **editable mode**, meaning:
  - Changes to the source code are immediately reflected
  - The package can be imported and run as an installed module from the environment

We can now run eg.
```bash
uv run pytest
uv run mypy src
uv run python -m ml_project_foundations.some_module
uv run python -m ml_project_foundations.some_file
```

## Project Goals

This project aims to:

- Showcase a **typical ML project layout**, including project structure, tooling, and best practices
- Explain the **technologies and packages** used, what problems they solve, and how they fit together
- Serve as a long-term **reference for learning, experimentation, and future projects**

## Why Structure Matters

Every serious ML project needs solid infrastructure. This includes:

- Clear **project and folder structure**
- Predictable **Python imports and packaging**
- Reproducible **Python environments and dependency management**
- Consistent **code formatting and linting**
- Continuous Integration (CI) via GitHub Actions, i.e. automated **building and testing**

- CD if used(might add for later)

This repository is designed to demonstrate how all of these pieces work together in a realistic setup.

## Project Structure

The project follows a `src/`-based layout:


<img
  src="https://github.com/user-attachments/assets/38b78670-6a3f-4747-95e3-c77b86006d73"
  alt="image"
  width="250"
  style="display: block; margin-left: 2;"
/>


Using a `src/` directory ensures:

- Predictable and explicit imports
- Tests import the package the same way it will be used in production
- Fewer issues with accidental relative imports (e.g. when running `pytest`)

The structure will evolve as the project grows, with each major folder introduced and explained when needed.

## Environment & Dependency Management

This project uses **uv** for:

- Python version management
- Virtual environment creation
- Dependency installation
- Packaging the project as a Python module

`uv` is a modern, fast alternative to traditional setups like `venv` + `pip`, keeping environment and dependency management under one umbrella.

## Code Quality & Standards

To enforce consistent and high-quality code, the project uses:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for static type checking
- **pre-commit** hooks to enforce formatting, linting, and type checks on every commit

This ensures that code style, correctness, and type safety are handled automatically.

## Type Checking

Static type checking is done using **mypy**.

- Helps catch bugs early by validating type correctness
- Improves code readability and self-documentation
- Makes refactoring safer as the project grows

Type checking configuration is managed via `pyproject.toml` and enforced both locally and in CI.

## Testing

Testing is done using **pytest**.

- All tests must pass before a pull request can be merged
- Tests are executed in an isolated environment
- Failures block merging to the main branch

## Continuous Integration (CI)

This project uses **GitHub Actions** for Continuous Integration.

On every pull request, the CI pipeline:

- Sets up a clean Python environment
- Installs dependencies using `uv`
- Runs:
  - Code formatting checks
  - Linting
  - Static type checking with `mypy`
  - Tests with `pytest`

This ensures that all code merged into the repository meets the same quality and correctness standards.
When using astral-sh/setup-uv in GitHub Actions, uv automatically persists its internal cache between workflow runs.

## Configuration

Most project configuration lives in **`pyproject.toml`**, including:

- Project packaging configuration
- Dependency definitions
- Tooling configuration (uv, black, ruff, mypy, pytest, etc.)

Centralizing configuration keeps the project easier to reason about and maintain.


## Development Tooling & Project Hygiene

This project prioritizes **consistency and reproducibility** over editor-specific
behavior.

All formatting, linting, and type checking are enforced at **commit time** and in
**CI**, ensuring identical results regardless of editor or local configuration.

### Repository Hygiene

Installing dependencies and running Python code produces various generated
artifacts. These files are expected, but they should **not** be committed to the
repository.

The following entries are included in `.gitignore` to prevent repository bloat:

```gitignore
__pycache__/
*.pyc
*.egg-info/
```

### Editor Setup (VSCode)

VSCode is my current choice of editor. A typical setup for this kind of projects.

Useful extensions include:

- **Python** (official Microsoft extension)
- **GitLens** (optional, for Git insights)

Formatting, linting, and type checking are intentionally **not** handled by editor
extensions. Instead, they are enforced via:

- `pre-commit` hooks
- GitHub Actions (CI)

This avoids editor-specific discrepancies and ensures a single source of truth.

### Editor Hygiene

When installing dependencies or running Python code, various generated artifacts
are created (e.g. `__pycache__`, `.egg-info` directories and *.pyc files).

These files are expected but should not be interacted with directly.

To reduce noise in the editor, the following exclusions are configured in
`.vscode/settings.json`:

```json
{
  "files.exclude": {
    "**/.git": true,
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/*.egg-info": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true
  }
}
```

## Version Control & Branching Strategy

This repository follows a simple and opinionated Git workflow to keep the history
clean and the `main` branch stable.

Workflows and conventions may vary between teams and projects. The choices here
reflect a reasonable default rather than a strict prescription.

Repository rules and branch protections are configured directly via the GitHub
web interface.

### Main Branch Protection

The `main` branch is protected and represents the current stable state of the
project.

Direct pushes to `main` are disabled. All changes must go through a pull request
(PR).

Branch protection rules enforce:

- All CI checks passing (tests, linting, type checking)
- No force-pushes to `main`
- No direct commits to `main`

This ensures that every change merged into `main` meets the same quality standards.

### Pull Requests

All development is done on feature branches.

Pull requests are used to:

- Run the full CI pipeline in a clean environment
- Review changes before merging
- Provide a clear audit trail of why changes were made

Once a pull request is merged, the feature branch is deleted to keep the repository
tidy and avoid stale branches.

### Rulesets

Repository rulesets are used to centrally define and enforce:

- Branch protection rules
- Required status checks(eg. passing pytest)
- Merge requirements

Using rulesets ensures consistent enforcement across branches and avoids relying
on individual developer discipline.






