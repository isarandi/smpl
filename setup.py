from setuptools import setup

setup(
    name='smpl',
    version='0.1.0',
    author='István Sárándi',
    author_email='istvan.sarandi@uni-tuebingen.de',
    packages=['smpl'],
    scripts=[],
    description='An implementation of the SMPL body model for NumPy, PyTorch and TensorFlow',
    python_requires='>=3.6',
    install_requires=['numpy'],
    extras_require={
        'tensorflow': ['tensorflow'],
        'pytorch': ['torch'],
    },
)
