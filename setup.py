from setuptools import find_packages, setup

setup(
    name='zunis',
    packages=find_packages(where='zunis_lib'),
    package_dir={'': 'zunis_lib'},
    install_requires=[
        "numpy == 1.19.1",
        "pandas == 1.1.0",
        "torch == 1.6.0",
        "abc_property == 1.0",
        "dictwrapper == 1.3",
        "ruamel.yaml <= 0.16.12"
    ],
    version='0.3',
    description='Neural Importance Sampling',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Nicolas Deutschmann',
    author_email="nicolas.deutschmann@gmail.com",
    url="https://zunis.readthedocs.io",
    download_url="https://github.com/ndeutschmann/zunis",
    license='MIT',
    include_package_data = True
)
