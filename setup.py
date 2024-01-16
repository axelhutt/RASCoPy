from setuptools import setup, find_packages

setup(
    name='RASCoPy',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy','matplotlib','torchvision','torch','seaborn','plotly','scipy'],
)