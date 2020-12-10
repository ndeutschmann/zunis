from setuptools import find_packages, setup

setup(
    name='zunis',
    packages=find_packages(),
    install_requires=[
        "numpy == 1.19.1",
        "pandas == 1.1.0",
        "torch == 1.6.0",
        "better-abc @ git+https://github.com/ndeutschmann/better_abc@6b393a2ae82da68e85f9a960eecbbb98bfd8d143",
    ],
    version='0',
    description='Neural Importance Sampling',
    long_description=open("../README.md").read(),
    author='Nicolas Deutschmann',
    author_email="nicolas.deutschmann@gmail.com",
    url="https://ndeutschmann.github.io/zunis/",
    download_url="https://github.com/ndetschmann/zunis",
    license='MIT',
)
