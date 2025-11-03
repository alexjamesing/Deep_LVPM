Continuous Integration
======================

The repository ships with a GitHub Actions workflow that exercises the
``deep-lvpm`` installation extras across multiple operating systems.
This page explains how Actions work in general, what the workflow does,
and how to adapt it to your needs.

How GitHub Actions run
----------------------

GitHub Actions executes YAML workflows stored in
``.github/workflows/``. Each workflow declares the events that trigger
it via the ``on`` key. In this project the installation matrix
responds to:

* ``push`` – every branch push runs the jobs.
* ``pull_request`` – opening or updating a pull request schedules the
  matrix so you get feedback before merging.
* ``workflow_dispatch`` – you can launch a manual run from the GitHub
  UI ("Run workflow" button).

When an event fires, GitHub provisions the appropriate virtual machine
for every job in the workflow and executes each ``steps`` block in
order.

What the installation matrix checks
-----------------------------------

The ``Validate installation extras`` workflow lives at
``.github/workflows/install-matrix.yml`` and currently defines a
single ``install`` job that fans out over a matrix of operating
systems, Python versions, and Deep LVPM extras. For every entry the job
performs these actions:

1. **Checkout** – fetch the repository contents so the job has access
   to ``setup.py`` and the package source.
2. **Set up Python** – install the requested Python version (3.10 by
   default) on the runner.
3. **Upgrade pip** – keep the packaging toolchain current.
4. **Install optional CUDA wheels** – for the ``torch-gpu`` extra the
   job preinstalls CUDA-enabled PyTorch wheels from the official
   PyTorch index.
5. **Install Deep LVPM** – run ``pip install .[extra]`` to validate that
   dependencies resolve correctly.
6. **Verify imports** – import ``deep_lvpm`` and the backend module
   implied by the extra (TensorFlow or PyTorch) to ensure both load
   successfully.

Thanks to the matrix, these steps execute in parallel for:

* Linux runners with TensorFlow and PyTorch (CPU and GPU variants).
* macOS runners that target Apple Silicon backends.
* Windows runners that exercise the CPU wheels.

Monitoring and rerunning jobs
-----------------------------

Every run shows up in the **Actions** tab of your GitHub repository.
Selecting a run reveals the per-matrix job logs. Failed jobs display a
red X; click into the job to review error messages. You can rerun
individual jobs or the entire workflow with the **Re-run jobs** button
once you have pushed a fix.

Customising the matrix
----------------------

Update the ``matrix.include`` section in
``install-matrix.yml`` to add or remove platforms and extras. Common
changes include:

* Testing additional Python versions by adding new entries with a
  different ``python-version`` field.
* Narrowing the matrix during early development (for example, removing
  GPU entries to shorten runtime) and reintroducing them once the
  workflow stabilises.
* Adding smoke-test commands after the import check—such as running a
  tutorial script—to extend coverage beyond installation.

Commit your changes to ``.github/workflows/install-matrix.yml`` and
push them to GitHub. The next ``push`` or ``pull_request`` event will
pick up the new configuration automatically.
