from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='vae4dag',
      py_modules=['vae4dag'],
      install_requires=[
          'torch==v1.4.0',
          'numpy',
          'tqdm',
          'networkx',
          'scipy'
      ],
)
