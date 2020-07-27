from setuptools import find_packages, setup


setup(
   name='contextual_lenses',
   version='1.0',
   description='Protein contextual lenses.',
   author='Amir Shanehsazzadeh',
   author_email='amirshanehsaz@google.com',
   packages=find_packages(exclude=('google-research', 'docs')),
   install_requires=[
   "jax",
   "jaxlib",
   "flax",
   "tensorflow",
   "pandas",
   "numpy"],
)
