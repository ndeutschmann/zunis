from setuptools import find_packages, setup

setup(
    name='zunis',
    packages=find_packages(),
    install_requires=[
        "numpy == 1.19.1",
        "pandas == 1.1.0",
        "torch == 1.6.0",
        "abc_property == 1.0",
        "dictwrapper == 1.3"
    ],
    version='0.2rc3',
    description='Neural Importance Sampling',
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    author='Nicolas Deutschmann',
    author_email="nicolas.deutschmann@gmail.com",
    url="https://ndeutschmann.github.io/zunis/",
    download_url="https://github.com/ndetschmann/zunis",
    license='MIT',
    include_package_data = True
)
