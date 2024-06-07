from setuptools import setup, find_packages

setup(
    name='ccHBGF',
    version='0.0.2',
    description='ccHBGF: Consensus Clustering using Hybrid Bipartite Graph Formulation (HBGF)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='E. H. von Rein',
    url='https://github.com/ehvr20/ccHBGF',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ],
    python_requires='>=3.6',
)