from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='multidag',
      py_modules=['multidag'],
      install_requires=[
          'torch>=v1.8.0',
          'numpy',
          'tqdm',
          'networkx',
          'scipy'
      ],
)
