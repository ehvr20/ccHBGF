from setuptools import setup, find_packages

setup(
    name='ccHBGF',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    author='E. H. von Rein',
    description='A consensus clustering algorithm using Hybrid Bipartite Graph Formulation (HBGF)',
    url='https://github.com/ehvr20/ccHBGF',
)
