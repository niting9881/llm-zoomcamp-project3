Apache MXNet (Incubating) Python Package
========================================
[Apache MXNet](http://beta.mxnet.io) is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to mix the flavours of deep learning programs together to maximize the efficiency and your productivity.

For feature requests on the PyPI package, suggestions, and issue reports, create an issue by clicking [here](https://github.com/apache/incubator-mxnet/issues/new).
Prerequisites
-------------
This package supports Linux, Mac OSX, and Windows platforms. You may also want to check:

- [mxnet-cu112](https://pypi.python.org/pypi/mxnet-cu112/) with CUDA-11.2 support.
- [mxnet-cu110](https://pypi.python.org/pypi/mxnet-cu110/) with CUDA-11.0 support.
- [mxnet-cu102](https://pypi.python.org/pypi/mxnet-cu102/) with CUDA-10.2 support.
- [mxnet-cu101](https://pypi.python.org/pypi/mxnet-cu101/) with CUDA-10.1 support.
- [mxnet-cu100](https://pypi.python.org/pypi/mxnet-cu100/) with CUDA-10.0 support.
- [mxnet-native](https://pypi.python.org/pypi/mxnet-native/) CPU variant without MKLDNN.

To use this package on Linux you need the `libquadmath.so.0` shared library. On
Debian based systems, including Ubuntu, run `sudo apt install libquadmath0` to
install the shared library. On RHEL based systems, including CentOS, run `sudo
yum install libquadmath` to install the shared library. As `libquadmath.so.0` is
a GPL library and MXNet part of the Apache Software Foundation, MXNet must not
redistribute `libquadmath.so.0` as part of the Pypi package and users must
manually install it.

To install for other platforms (e.g. Windows, Raspberry Pi/ARM) or other versions, check [Installing MXNet](https://mxnet.apache.org/versions/master) for instructions on building from source.

Installation
------------
To install, use:
```bash
pip install mxnet
```

