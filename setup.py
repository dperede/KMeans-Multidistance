from distutils.core import setup

long_desc = """
A library that allows to use the K-Means algorithm using having the possibility to choose between multiple metrics to calculate the distance between the data points in the feature space.
"""

setup(
    name='KMeans-Multidistance',
    version='1.0.1',
    description='K-Means implementation with multiple distance choices',
    long_description=long_desc,
    author='David Perez Delgado',
    author_email='cathan.89@gmail.com',
    url='https://github.com/dperede/KMeans-Multidistance',
    download_url='https://github.com/dperede/KMeans-Multidistance/archive/v1.0.1.tar.gz',
    install_requires=[
      'numpy==1.15.4',
      'matplotlib>=3.0.2',
      'scikit-learn>=0.20.1'
    ],
    packages=['KMeans_Multidistance']
)
