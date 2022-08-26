from distutils.core import setup

setup(
    name='tc1-stablity',
    version='0',
    packages=[
        'tc1-stability',
    ],
    install_requires=[
        'openmdao',
        'csdl',
        'csdl_om',
        ],
)