from setuptools import find_packages
from setuptools import setup


setup(
    name='pytorch-serverless',
    version='0.0.1',
    description='PyTorch Serverless production API (w/ AWS Lambda)',
    url='https://github.com/alecrubin/pytorch-serverless',
    author='Alec Rubin',
    keywords='PyTorch, Serverless, AWS Lambda, API',
    packages=find_packages(exclude=["tests.*", "tests"])
)
