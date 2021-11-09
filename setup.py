import setuptools

with open("requirements.txt", encoding="utf8") as f:
    requirements = f.readlines()

setuptools.setup(
    name="libtwenty",
    version="1.0.0",
    author="Z1R343L",
    description="2048 lib",
    url="https://github.com/z1r343l",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    include_package_data=True,
)
