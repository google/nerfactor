.. xiuminglib documentation master file, created by
   sphinx-quickstart on Mon Mar 25 15:24:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xiuminglib's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

xiuminglib includes daily classes and functions that are useful for my computer
vision/graphics research. Noteworthily, it contains useful functions for 3D
modeling and rendering with Blender.

To get a sense of what it is capable of, scroll to the bottom for a tree of
its modules and functions. The source code is available in
`the repo <https://github.com/xiumingzhang/xiuminglib>`_. For issues or
questions, please open an issue there.


Installation
============

First, clone the repo. and add it to your ``PYTHONPATH``:

.. code-block:: bash

    cd <your_local_dir>
    git clone https://github.com/xiumingzhang/xiuminglib.git
    export PYTHONPATH="<your_local_dir>/xiuminglib/":"$PYTHONPATH"

Install the dependencies automatically with Conda: simply create an environment
with all the dependencies by running:

.. code-block:: bash

    cd <your_local_dir>/xiuminglib/
    conda env create -f environment.yml
    conda activate xiuminglib

If you do not need Blender functionalities, you are all set. Otherwise, you
need to (manually) install Blender as a Python module, as instructed below.

If you want to avoid Conda environments, also see the dependencies below and
manually install each your own way.

(Optional) Manual Dependency Installation
-----------------------------------------

The library uses "on-demand" imports whenever possible, so that it will not fail
on imports that you do not need.

If you want Blender, you need to install it as a Python module manually
(regardless of using Conda or not):

    Blender 2.79
        Note this is different from installing Blender as an application,
        which has Python bundled. Rather, this is installing Blender as a
        Python module: you have succeeded if you find ``bpy.so`` in the
        build's bin folder and can ``import bpy`` in your Python (not the
        Blender-bundled Python) after you add it to your ``PYTHONPATH``.

        Ubuntu
            I did this "the hard way": first building all dependencies from
            source manually, and then
            `building Blender from source <https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu>`_
            with ``-DWITH_PYTHON_MODULE=ON`` for CMake, primarily because I
            wanted to build to an NFS location so that a cluster of machines on
            the NFS can all use the build.

            If you only need Blender on a local machine, for which you can
            ``sudo``, then dependency installations are almost automatic -- just
            run ``install_deps.sh``, although when I did this, I had to
            ``skip-osl`` to complete the run, for some reason I did not take
            time to find out.

            Blender 2.80 made some API changes that are incompatible with this
            library, so please make sure after ``git clone``, you check out
            `the correct tag <https://git.blender.org/gitweb/gitweb.cgi/blender.git/tag/refs/tags/v2.79b>`_
            with ``git checkout v2.79b``, followed by ``git submodule update``
            to ensure the submodules are of the correct versions.

            If ``import bpy`` throws ``Segmentation fault``, try again with
            Python 3.6.3.

        macOS
            `This instruction <https://wiki.blender.org/wiki/Building_Blender/Mac>`_
            was not very helpful, so below documents each step I took to
            finally get it working (though with some non-fatal warnings).

            First, install Xcode 9.4 to build against the old ``libstdc++``
            (instead of
            `Xcode 10+ that forces the use of the newer <https://stackoverflow.com/a/42034101/2106753>`_
            ``libc++``). Then, ``brew install`` CMake.

            Install
            `Python Framework 3.6.3 <https://www.python.org/ftp/python/3.6.3/python-3.6.3-macosx10.6.pkg>`_.
            I tried to use an Anaconda Python, but to no avail.

            Clone the Blender repo., check out v2.79b, and make sure submodules
            are consistent.

            .. code-block:: bash

                mkdir ~/blender-git && cd ~/blender-git
                git clone https://git.blender.org/blender.git && cd blender
                git checkout v2.79b # may also work: git reset --hard v2.79b
                git submodule update --init --recursive
                git submodule foreach git checkout master
                git submodule foreach git pull --rebase origin master

            Download the pre-built libraries, and move them to the correct
            place.

            .. code-block:: bash

                cd ~/blender-git
                svn export https://svn.blender.org/svnroot/bf-blender/tags/blender-2.79-release/lib/darwin-9.x.universal/
                mkdir lib && mv darwin-9.x.universal lib/

            Edit
            ``~/blender-git/blender/build_files/cmake/platform/platform_apple.cmake``
            to replace ``set(PYTHON_VERSION 3.5)`` with
            ``set(PYTHON_VERSION 3.6)``.

            Make ``bpy.so`` by running
            ``cd ~/blender-git/blender && make bpy``. You *may* also need
            ``cd ~/blender-git/build_darwin_bpy && make install``. Upon
            success, ``bpy.so`` is in ``~/blender-git/build_darwin_bpy/bin/``,
            and so is ``2.79/``.

            For ``scripts/modules`` to be found during import, do

            .. code-block:: bash

                mkdir ~/blender-git/build_darwin_bpy/Resources
                cp -r ~/blender-git/build_darwin_bpy/bin/2.79 ~/blender-git/build_darwin_bpy/Resources/

            Add the bin folder to ``PYTHONPATH`` with
            ``export PYTHONPATH="~/blender-git/build_darwin_bpy/bin/":"$PYTHONPATH"``.

            Verify your success with

            .. code-block:: bash

                /Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6 \
                    -c 'import bpy; bpy.ops.render.render(write_still=True)'

            but expect the aforementioned "non-fatal warnings":

            .. code-block:: bash

                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                ModuleNotFoundError: No module named 'io_scene_3ds'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_scene_fbx/__init__.py", line 52, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_anim_bvh/__init__.py", line 49, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_mesh_ply/__init__.py", line 56, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_scene_obj/__init__.py", line 48, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_scene_x3d/__init__.py", line 48, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 331, in enable
                    mod = __import__(module_name)
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_mesh_stl/__init__.py", line 66, in <module>
                    from bpy_extras.io_utils import (
                ImportError: cannot import name 'orientation_helper'
                Exception in module register(): '/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_curve_svg/__init__.py'
                Traceback (most recent call last):
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/modules/addon_utils.py", line 350, in enable
                    mod.register()
                  File "/Users/xiuming/blender-git/build_darwin_bpy/Resources/2.79/scripts/addons/io_curve_svg/__init__.py", line 70, in register
                    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
                AttributeError: 'RNA_Types' object has no attribute 'TOPBAR_MT_file_import'

Only if you are not automatically installing the dependencies, you need to
manually install whatever you need:

    NumPy
        `The package for scientific computing <https://numpy.org/>`_ that should
        be already available as part of your Python distribution.

    SciPy
        `The scientific computing ecosystem <https://www.scipy.org/>`_ that may
        or may not be pre-installed already.

    Matplotlib 2.0.2
        Some functions are known to be buggy with 3.0.0.

    tqdm
        `A progress bar <https://tqdm.github.io/>`_.

    Pillow
        `The friendly PIL fork <https://pillow.readthedocs.io/en/stable/>`_.

    OpenCV
        ``pip install opencv-python`` seems to work better than
        ``conda install``. If any ``lib*.so*`` is missing at runtime (which
        happens often with ``conda install``), the easiest fix is to install
        the missing library to the same environment, maybe followed by some
        symlinking (like linking ``libjasper.so`` to ``libjasper.so.1``)
        inside ``<python_dir>/envs/<env_name>/lib``. This may be cleaner
        and easier than ``apt-get``, which may break other things and usually
        requires ``sudo``.

    Trimesh
        See
        `their installation guide <https://github.com/mikedh/trimesh/blob/master/docs/install.rst>`_.

    TensorFlow
        See `this installation guide <https://www.tensorflow.org/install>`_.

    IPython
        This is required only for debugging purposes (e.g., inserting
        breakpoints with its ``embed()``). Skip it if you do not care.

    Sphinx 2.0.1 & RTD Theme
        These are required only by documentation building. Feel free to skip
        them if you do not care. The RTD theme package is called
        ``sphinx_rtd_theme``.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: modules.rst
