import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='mogp_emulator',
      version='0.1',
      description='Tool for Multi-Output Gaussian Process Emulators',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='TBD',
      author='Eric Daub',
      author_email='edaub@turing.ac.uk',
      packages=setuptools.find_packages(),
      license=['TBD'],
      install_requires=['numpy', 'scipy', 'gp_emulator'])