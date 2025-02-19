# Contributing

Welcome to ``gReLU`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but [other kinds of contributions](https://opensource.guide/how-to-contribute) are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at [contribution-guide.org](https://www.contribution-guide.org/). Other resources are also
listed in the excellent [guide created by FreeCodeCamp](https://github.com/FreeCodeCamp/how-to-contribute) [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, [Python Software
Foundation's Code of Conduct](https://www.python.org/psf/conduct/) is a good reference in terms of behavior
guidelines.

## Issue Reports

### Check the issue tracker

If you experience bugs or general issues with ``grelu``, please have a look
on the [issue tracker](https://github.com/Genentech/gReLU/issues). Don't forget to include the closed issues in your search. Sometimes a solution was already reported, and the problem is considered **solved**.

### File an issue report
If you don't see anything useful in the previous issues, please feel free to fire an issue report.
New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.

## Code Contributions

### Package structure

The gReLU repository is organized as follows:

- src/: Contains the main source code for gReLU.
- tests/: Includes unit tests and integration tests.
- docs/: Houses the documentation files.
- CONTRIBUTING.md: The file you're reading now.
- README.md: Provides an overview of the project.
- LICENSE: Contains the terms under which the code can be used and distributed.

Check the module index at https://genentech.github.io/gReLU/py-modindex.html for further details of the API.

### Submit an issue

Before you work on any non-trivial code contribution it's best to first create
a report in the [issue tracker](https://github.com/Genentech/gReLU/issues) to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

### Create an environment

Before you start coding, we recommend creating an isolated [virtual
environment](https://realpython.com/python-virtual-environments-a-primer/) to avoid any problems with your installed Python packages.
This can easily be done via either [virtualenv](https://virtualenv.pypa.io/en/stable/)::

    virtualenv <PATH TO VENV>
    source <PATH TO VENV>/bin/activate

or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)::

    conda create -n grelu python=3 six virtualenv pytest pytest-cov
    conda activate grelu

### Clone the repository

1. Create an user account on [the repository service](https://github.com). if you do not already have one.
2. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on [the repository service](https://github.com/).
3. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/grelu.git
    cd grelu

4. You should run::

    pip install -U pip setuptools -e .

   to be able to import the package under development in the Python REPL.

5. Install [pre-commit](https://pre-commit.com/)::

    pip install pre-commit
    pre-commit install

   ``grelu`` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

### Implement your changes

1. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

2. Implement your changes on this branch.

3. If you make significant changes (not just a bugfix), don't forget to update any docstrings and unit tests that are affected. If you add new functions, modules, or classes, don't forget to write docstrings and unit tests for them.

5. Add yourself to the list of contributors in ``AUTHORS.rst``.

6. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   Please make sure to see the validation messages from [pre-commit](https://pre-commit.com/) and fix
   any eventual issues.
   This should automatically use [flake8](https://flake8.pycqa.org/en/stable/)/[black](https://pypi.org/project/black/) to check/fix the code style
   in a way that is compatible with the project.

      Moreover, writing a [descriptive commit message](https://chris.beams.io/posts/git-commit) is highly recommended.


7. Please check that your changes don't break any unit tests with::

    tox

   (after having installed [tox](https://tox.wiki/en/stable/) with ``pip install tox`` or ``pipx``).

   You can also use [tox](https://tox.wiki/en/stable/) to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

### Submit your contribution

1. If everything works fine, push your local branch to [the repository service](https://github.com/) with::

    git push -u origin my-feature

2. Go to the web page of your fork and click "Create pull request"
   to send your changes for review.

      Find more detailed information in [creating a PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). You might also want to open
      the PR as a draft first and mark it as ready for review after the feedbacks
      from the continuous integration (CI) system or any required fixes.


## Where to contribute new functionality

Look at the API reference (https://genentech.github.io/gReLU/autoapi/index.html) to see the modules and submodules available in gReLU. Clicking on individual modules revels a description of what kinds of functions the module should contain. This will help you find the appropriate location for new functions that you want to contribute. The descriptions also contain more detailed explanations of the expected structure of each module and how to contribute to it.

For example:

- Functions to read / write genomic data should be added to `grelu.io`. If there is an existing submodule for the relevant genomic data format, you should add your new function to the submodule, otherwise add it to `__init__.py` or create a new submodule.
- Functions to preprocess genomic data after it is loaded should be added to `grelu.data.preprocess`.
- New augmentation functions for training models should be added to `grelu.data.augment`.
- Functions to manipulate DNA sequences should be added to `grelu.sequence.utils`.
- Functions to score DNA sequences based on their content should be added to `grelu.transforms.seq_transforms`
- Functions to transform label values, e.g. scale or normalize sequencing coverage, should be added to `grelu.transforms.label_transforms`
- Functions to transform model predictions, e.g. compute the total predicted coverage over some sequence region, should be added to `grelu.transforms.prediction_transforms`.
- New loss functions should be added to `grelu.lightning.losses`.
- New types of positional encoding should be added to `grelu.model.position`.
- New types of model layers should be added to `grelu.model.layers`.

For more complex changes that may not fit clearly within the established package structure, we suggest raising an issue (see instructions above).

## Troubleshooting

The following tips can be used when facing problems to build or test the
package:

1. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

2. Sometimes [tox](https://tox.wiki/en/stable/) misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with [tox](https://tox.wiki/en/stable/), try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

3. Make sure to have a reliable [tox](https://tox.wiki/en/stable/) installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run::

```shell
    tox --version
    # OR
    which tox
```

   If you have trouble and are seeing weird errors upon running [tox](https://tox.wiki/en/stable/), you can
   also try to create a dedicated [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) with a [tox](https://tox.wiki/en/stable/) binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

4. [Pytest can drop you](https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest) in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


## Maintainer tasks

### Releases

If you are part of the group of maintainers and have correct user permissions
on [PyPI](https://pypi.org/), the following steps can be used to release a new version for
``grelu``:

1. Make sure all unit tests are successful.
2. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
3. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
4. Clean up the ``dist`` and ``build`` folders with ``tox -e clean``
   (or ``rm -rf dist build``)
   to avoid confusion with old builds and Sphinx docs.
5. Run ``tox -e build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
6. Run ``tox -e publish -- --repository pypi`` and check that everything was
   uploaded to PyPI_ correctly.


.. <-- start -->

.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/Genentech/gReLU/
.. _issue tracker: https://github.com/Genentech/gReLU/issues

.. <-- end -->
