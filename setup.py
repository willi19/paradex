from pathlib import Path
from setuptools import find_packages, setup

# Core requirements
core_requirements = [

]

setup(
    name="paradex",
    version="0.1",
    author="Mingi Choi",
    author_email="willi19@snu.ac.kr",
    url="",
    description="managing camera system of paradex",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["paradex.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=core_requirements,
)