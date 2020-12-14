import os
import setuptools

from distutils.extension import Extension
from distutils.command.build_ext import build_ext

import numpy as np

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
numpy_include = np.get_include()

ext = Extension("libgpgpu",
                sources=["mogp_gpu/src/gp_gpu.cu","mogp_gpu/src/cov_gpu.cu","mogp_gpu/src/util.cu"],
                library_dirs=[cuda_config["lib64"]],
                libraries=["cudart","cublas","cusolver"],
                runtime_library_dirs=[cuda_config["lib64"]],
                extra_compile_args={"gcc":["-std=c++11"],
                                    "nvcc": ["--compiler-options","-O3,-Wall,-shared,-std=c++11,-fPIC",
                                             "--generate-code","arch=compute_60,code=sm_60"
                                    ]},
                include_dirs=[numpy_include, cuda_config["include"],"mogp_gpu/src"])

setuptools.setup(name='mogp_emulator',
                 version='0.1',
                 description='Tool for Multi-Output Gaussian Process Emulators',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='TBD',
                 author='Eric Daub',
                 author_email='edaub@turing.ac.uk',
                 packages=setuptools.find_packages(),
                 license=['MIT'],
                 ext_modules=[ext],
                 cmdclass={"build_ext": custom_build_ext},
                 install_requires=['numpy', 'scipy'],
                 zip_safe=False

)
