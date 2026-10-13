# Build script for the symnmfmodule C extension.

from setuptools import Extension, setup

setup(
    name='symnmfmodule',
    version='0.1.0',
    description='C extension implementing the symNMF matrix goals',
    ext_modules=[
        # symnmf.c is listed too: the wrapper has no math of its own to link.
        Extension('symnmfmodule', sources=['symnmfmodule.c', 'symnmf.c'])
    ]
)
