import os
import setuptools

from distutils.extension import Extension
from distutils.command.build_ext import build_ext

import numpy as np

# version information
MAJOR = 0
MINOR = 6
MICRO = 0
PRERELEASE = 0
ISRELEASED = True
version = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

if not ISRELEASED:
    version += ".dev{}".format(PRERELEASE)

# write version information to file

def write_version_file(version):
    "writes version file that is read when importing version number"
    version_file = """'''
Version file automatically created by setup.py file
'''
version = '{}'
    """.format(version)

    with open("mogp_emulator/version.py", "w") as fh:
        fh.write(version_file)

write_version_file(version)

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_cuda_config():
    """
    Find the nvcc executable in the path, and
    return a dictionary with keys 'home', 'nvcc', 'include', 'lib64'.
    """
    path = os.environ["PATH"]
    exe = "nvcc"
    binpath = None
    for location in path.split(os.pathsep):
        if os.path.exists(os.path.join(location, exe)):
            binpath = os.path.abspath(os.path.join(location, exe))
            break
    if not binpath:
        print("Unable to find 'nvcc' in path")
        return {}
    home = os.path.dirname(os.path.dirname(binpath))
    include = os.path.join(home, "include")
    lib64 = os.path.join(home, "lib64")
    return {
        "home": home,
        "nvcc": binpath,
        "include": include,
        "lib64": lib64
    }

def find_dlib():
    """
    Find the dlib directory in LD_LIBRARY_PATH and return a 
    dict with keys "include" and "lib64" containing the appropriate paths.
    """
    path = os.environ["LD_LIBRARY_PATH"]
    locations = path.split(os.pathsep)
    for loc in ["/usr/lib","/usr/local/lib"]:
        if not loc in locations:
            locations.append(loc)
    base_dir = None
    lib_dir = None
    for location in locations:
        if len(location) ==0:
            continue
        else:
            try:
                if "libdlib.a" in os.listdir(location):
                    base_dir = os.path.dirname(location)
                    lib_dir = os.path.basename(location)
                    break
                # it is possible that we don't have the correct
                # directory in LD_LIBRARY_PATH, e.g. we have
                # /path/to/dlib/lib while the actual library is in
                # /path/to/dlib/lib64
                # Try to deal with that situation here:
                elif "dlib" in location:
                    base_dir = os.path.dirname(location)
                    for subdir in os.listdir(base_dir):
                        if "libdlib.a" in os.listdir(os.path.join(base_dir, subdir)):
                            lib_dir = subdir
                            break
            except(FileNotFoundError):
                pass
    if not base_dir and lib_dir:
        print("unable to find dlib in LD_LIBRARY_PATH")
        return {}
    include = os.path.join(base_dir,"include")
    lib64 = os.path.join(base_dir, lib_dir)
    return {
        "include": include,
        "lib64": lib64
    }


def customize_compiler_for_nvcc(self):
    """
    Taken from SO
    https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code
    by Robert T. McGibbon.
    Modify the _compile of distutils extension builder.
    """
    # add .cu to list of extensions compile knows about
    self.src_extensions += ['.cu',".hpp"]
    # save references to original methods
    default_compiler_so = self.compiler_so
    super = self._compile
    # redefine _compile method.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use nvcc for .cu files
            cuda_config = get_cuda_config()
            if not "nvcc" in cuda_config.keys():
                return
            self.set_executable('compiler_so', cuda_config['nvcc'])
            # use subset of extra_postargs
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset default compiler_so
        self.compiler_so = default_compiler_so
    # replace the class default _compile method with this one
    self._compile = _compile

# Run the custom compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

cuda_config = get_cuda_config()

# we only want to add the mogp_gpu extension if we have cuda compiler
ext_modules = []
if len(cuda_config) > 0:
    import pybind11
    pybind_include = pybind11.get_include()
    numpy_include = np.get_include()
    dlib_location = find_dlib()
    dlib_dir = dlib_location["lib64"]
    dlib_include = dlib_location["include"]
    
    ext = Extension("libgpgpu",
                    sources=["mogp_gpu/src/bindings.cu",
                             "mogp_gpu/src/kernel.cu",
                             "mogp_gpu/src/util.cu"],
                    library_dirs=[cuda_config["lib64"], dlib_dir],
                    libraries=["cudart","cublas","cusolver","dlib","openblas"],
                    runtime_library_dirs=[cuda_config["lib64"]],
                    extra_compile_args={"gcc":["-std=c++14"],
                                    "nvcc": ["--compiler-options",
                                             "-O3,-Wall,-shared,-std=c++14,-fPIC,-fopenmp",
                                             "-arch=sm_60",
                                             "--generate-code","arch=compute_37,code=sm_37",
                                             "--generate-code","arch=compute_60,code=sm_60"
                                    ]},
                    extra_link_args=["-lgomp"],
                    include_dirs=[numpy_include, pybind_include, cuda_config["include"],"mogp_gpu/src", dlib_include])
    ext_modules.append(ext)

setuptools.setup(name='mogp_emulator',
      version=version,
      description='Tool for Multi-Output Gaussian Process Emulators',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://mogp-emulator.readthedocs.io/',
      project_urls={
                  "Bug Tracker": "https://github.com/alan-turing-institute/mogp-emulator/issues",
                  "Documentation": "https://mogp-emulator.readthedocs.io/",
                  "Source Code": "https://github.com/alan-turing-institute/mogp-emulator/",
              },
      author='Alan Turing Institute Research Engineering Group',
      author_email='edaub@turing.ac.uk',
      packages=setuptools.find_packages(),
      license='MIT',
      ext_modules=ext_modules,
      cmdclass={"build_ext": custom_build_ext},
      install_requires=['numpy', 'scipy', 'patsy'],
      zip_safe=False)
