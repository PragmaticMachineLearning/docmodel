from setuptools import setup 

setup(
        name = "docmodel",
        version = "1.0",
        install_requires = [line for line in open("requirements/requirements.txt", "r", encoding="utf-8")],
        license = "MPL-2.0"
        )
