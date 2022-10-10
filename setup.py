from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gwu_nn',
      version='0.1',
      description='Neural Network library for the George Washington University',
      url='https://gitlab.com/gwu_intro_nn/gwu_nn',
      author='Joel Klein',
      author_email='jdk51405@gmail.com',
      long_description=long_description,
      license='MIT',
      classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      packages=['gwu_nn'],
      install_requires=[
        'numpy',
      ],
      zip_safe=False)