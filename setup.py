from setuptools import setup

setup(
   name='contextual_lenses',
   version='1.0',
   description='Protein contextual lenses.',
   author='Amir Shanehsazzadeh',
   author_email='amirshanehsaz@google.com',
   packages=['contextual_lenses'],
   install_requires=['jax', 'jaxlib', 'flax'],
)