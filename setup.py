from setuptools import setup


with open('requirements.txt', 'r') as f:
	install_requires = f.read()

setup(
   name='contextual_lenses',
   version='1.0',
   description='Protein contextual lenses.',
   author='Amir Shanehsazzadeh',
   author_email='amirshanehsaz@google.com',
   packages=['contextual_lenses'],
   install_requires=install_requires,
)