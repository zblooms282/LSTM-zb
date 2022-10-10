from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gwu_nn',
      version='0.1.3',
      description='Neural Network library for the George Washington University',
      url='https://gwu-nn.readthedocs.io/en/latest/index.html',
      author='Joel Klein',
      author_email='jdk51405@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT',
      classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      packages=['gwu_nn'],
      install_requires=[
        'numpy >= 1.20',
      ],
      zip_safe=False,
      python_requires='>=3.7'
)