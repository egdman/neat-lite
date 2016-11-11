from setuptools import setup, find_packages


setup(
	name='neat-lite',
	version='0.0.1.dev',
	description='A lightweight library that implements NeuroEvolution through Augmenting Topologies.',
	url='https://github.com/egdman/neat',
	author='Dmitry Egorov',
	author_email='egdman90@gmail.com',
	classifiers = [
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7',
		'Topic :: Scientific/Engineering :: Artificial Intelligence'
	],
	license='MIT',
	install_requires=['numpy']

)