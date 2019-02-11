from distutils.core import setup

long_desc = """
A library that allows to use the K-Means algorithm using having the possibility to choose between multiple metrics to calculate the distance between the data points in the feature space.
"""

setup(
    name='KMeans-Multidistance',
    version='0.0.94',
    description='K-Means implementation with multiple distance choices',
    long_description=long_desc,
    author='David Perez Delgado',
    author_email='cathan.89@gmail.com',
    url='https://github.com/dperede/KMeans-Multidistance',
    download_url='https://github.com/dperede/KMeans-Multidistance.git',
    install_requires=[
      'numpy==1.15.4',
      'matplotlib>=3.0.2',
      'scikit-learn>=0.21.1'
    ],
    packages=['KMeans_Multidistance']
)