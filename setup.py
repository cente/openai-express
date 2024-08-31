from setuptools import setup, find_packages

setup(
    name='openai-express',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai',  # or any other dependencies your project needs
    ],
    entry_points={
        'console_scripts': [
            'openai-express=openai_express.main:cli_interface',
        ],
    },
    include_package_data=True,
    python_requires='>=3.6',
)