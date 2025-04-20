import pathlib
import pkg_resources
from setuptools import find_packages, setup


def _read_install_requires():
    with pathlib.Path('requirements.txt').open() as fp:
        packages = [str(requirement) for requirement in pkg_resources.parse_requirements(fp)]
        return packages


setup(
    name='lift3d',
    version='1.0',
    author='CSCSX',
    author_email='cscsxiang@gmail.com',
    url='https://github.com/PKU-HMI-Lab/EAI-Representation-Learning/tree/lift3d-merge',
    description='Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation',
    long_description=pathlib.Path('README.md').open().read(),
    long_description_content_type='text/markdown',
    keywords=[
        'Robotics',
        'Embodied Intelligence',
        'Representation Learning',
    ],
    license='MIT License',
    packages=find_packages(include='lift3d.*'),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    python_requires='>=3.9',
)