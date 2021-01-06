#!/usr/bin/env python
import os
import re
from io import open
from setuptools import setup, find_packages


def get_property(property, package):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(property),
        open(package + '/__init__.py').read(),
    )
    return result.group(1)


from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf8') as f:
    long_description = f.read()

# Scripts
scripts = []
for dirname, dirnames, filenames in os.walk('scripts'):
    for filename in filenames:
        if filename.endswith('.py'):
            scripts.append(os.path.join(dirname, filename))

setup(
    name='semiclassical',
    version=get_property('__version__', 'semiclassical'),
    description='Internal conversion rates from semiclassical molecular dynamics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/humeniuka/semiclassical',
    author='Alexander Humeniuk',
    author_email='alexander.humeniuk@gmail.com',
    license='LICENSE.txt',
    packages=find_packages(),
    install_requires=['numpy >= 1.19.0', 'ase >= 3.16.2', 'torch >= 1.8.0.dev20201208'],
    entry_points={
        'console_scripts': ['semi=semiclassical.cli:main']
    },
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
)
