import setuptools

# version information
MAJOR = 0
MINOR = 3
MICRO = 0
PRERELEASE = 14
ISRELEASED = False
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

setuptools.setup(name='mogp_emulator',
      version=version,
      description='Tool for Multi-Output Gaussian Process Emulators',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://mogp-emulator.readthedocs.io/',
      project_urls={
                  "Bug Tracker": "https://github.com/alan-turing-institute/mogp_emulator/issues",
                  "Documentation": "https://mogp-emulator.readthedocs.io/",
                  "Source Code": "https://github.com/alan-turing-institute/mogp_emulator/",
              },
      author='Alan Turing Institute Research Engineering Group',
      author_email='edaub@turing.ac.uk',
      packages=setuptools.find_packages(),
      license=['MIT'],
      install_requires=['numpy', 'scipy'])