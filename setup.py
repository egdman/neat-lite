from setuptools import setup
from neat import __version__

setup(
	name='neat-lite',
	version=__version__,
	description='A dependency-free Python 3 library that implements NeuroEvolution through Augmenting Topologies.',
	url='https://github.com/egdman/neat-lite',
	author='Dmitry Egorov',
	author_email='egdman90@gmail.com',
	classifiers = [
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3.7',
		'Topic :: Scientific/Engineering :: Artificial Intelligence'
	],
	license='MIT',
	packages=['neat'],
)
