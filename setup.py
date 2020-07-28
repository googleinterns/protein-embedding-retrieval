from setuptools import find_packages, setup

# This follows the style of Jaxlib installation here:
# https://github.com/google/jax#pip-installation
PYTHON_VERSION = "cp37"
CUDA_VERSION = "cuda101" # alternatives: cuda90, cuda92, cuda100, cuda101
PLATFORM = "linux_x86_64" # alternatives: linux_x86_64
BASE_URL = "https://storage.googleapis.com/jax-releases"


def jax_artifact(version, gpu=False):
    if gpu:
        prefix = f"{BASE_URL}/{CUDA_VERSION}/jaxlib"
        wheel_suffix = f"{PYTHON_VERSION}-none-{PLATFORM}.whl"
        location = f"{prefix}-{version}-{wheel_suffix}"
        return f"jaxlib @ {location}"
    return f"jaxlib=={version}"

def readme():
    try:
        with open('README.md') as rf:
            return rf.read()
    except FileNotFoundError:
        return None

JAXLIB_VERSION = "0.1.43"
JAX_VERSION = "0.1.62"

REQUIRED_PACKAGES = [
                     "tensorflow",
                     "pandas",
                     "numpy",
                     f"jax=={JAX_VERSION}",
                     "flax"
                     ]

setup(name='contextual_lenses',
      version='1.0',
      description='Protein contextual lenses.',
      long_description=readme(),
      author='Amir Shanehsazzadeh',
      author_email='amirshanehsaz@google.com',
      packages=find_packages(exclude=('docs')),
      install_requires=REQUIRED_PACKAGES,
      extras_require={
      "cpu": [jax_artifact(JAXLIB_VERSION, gpu=False)],
      "gpu": [jax_artifact(JAXLIB_VERSION, gpu=True)],
      })
