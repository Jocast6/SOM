from setuptools import setup, find_packages

setup(
    name='HexSOM',                       # Name of your package
    version='0.6.0',                     # Version of the package
    packages=find_packages(),            # Automatically find your packages
    install_requires=[                   # List of dependencies (if any)
        'numpy', 'scipy'
    ],
    long_description=open('README.md').read(),   # Long description from README
    long_description_content_type='text/markdown',  # Format for long description
    author='Joel G. Castro',                  # Your name
    author_email='joey-92@live.com',  # Your email
    description='Short description of your package',
    url='https://github.com/yourusername/your-package',  # Link to your package’s homepage
    classifiers=[                        # Classifiers help people find your package
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)