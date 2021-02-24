from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='gan4dag',
      py_modules=['gan4dag'],
      install_requires=[
          'torch',
          'numpy'
      ],
)
