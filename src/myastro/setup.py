from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="kepler_u", sources=["kepler_u.pyx"])
setup(ext_modules=cythonize(ext))
