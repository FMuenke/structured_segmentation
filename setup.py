# python setup.py bdist_wheel
# !/usr/bin/env python

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
      "scikit-learn",
      "numpy",
      "scikit-image",
      "opencv-python",
      "scipy",
      "joblib",
      "tqdm",
      "umap-learn",
]

setup(name='structured_segmentation',
      version='1.0.0',
      description='This python package implements structured segmentation algorithms for semantic segmentation',
      author='Friedrich Muenke',
      author_email='friedrich.muenke@me.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=REQUIRED_PACKAGES,)
