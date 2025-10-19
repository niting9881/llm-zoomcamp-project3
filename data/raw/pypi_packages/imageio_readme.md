
.. image:: https://github.com/imageio/imageio/workflows/CI/badge.svg
    :target: https://github.com/imageio/imageio/actions


Imageio is a Python library that provides an easy interface to read and
write a wide range of image data, including animated images, volumetric
data, and scientific formats. It is cross-platform, runs on Python 3.9+,
and is easy to install.

Main website: https://imageio.readthedocs.io/


Release notes: https://github.com/imageio/imageio/blob/master/CHANGELOG.md

Example:

.. code-block:: python

    >>> import imageio
    >>> im = imageio.imread('imageio:astronaut.png')
    >>> im.shape  # im is a numpy array
    (512, 512, 3)
    >>> imageio.imwrite('astronaut-gray.jpg', im[:, :, 0])

See the `API Reference <https://imageio.readthedocs.io/en/stable/reference/index.html>`_
or `examples <https://imageio.readthedocs.io/en/stable/examples.html>`_
for more information.
