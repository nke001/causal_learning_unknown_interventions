#
# Imports
#
import os
from   setuptools import setup, find_packages, Extension

#
# Perform setup.
#
setup(
    name                 = "causal",
    version              = "0.0.0",
    author               = "Anonymous Anonymous",
    author_email         = "anonymous@anonymous.com",
    description          = "Causal Meta-Learning from Interventions",
    classifiers          = [
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires      = '>=3.6',
    install_requires     = [
        "numpy>=1.10",
        "torch>=1.1",
        "nauka>=0.0.11",
        "Pillow>=4.0.0",
    ],
    packages             = find_packages('.'),
    ext_modules          = [
        Extension(
            name               = "causal._causal",
            sources            = [os.path.join("causal", "_causal.c")],
            extra_compile_args = ["-O3", "-march=native",
                                  "-msse", "-msse2", "-mavx", "-mavx2", "-mfma"],
        )
    ],
    zip_safe             = False,
)
