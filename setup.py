from pathlib import Path
from setuptools import find_packages, setup

# Core requirements
core_requirements = [

]

extras_require = {
    # Aravis itself and PyGObject are Ubuntu packages. pyroute2 is used only
    # by capture-PC ForceIP recovery, so keep it out of the main-PC install.
    "aravis": ["pyroute2>=0.7"],
}

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
    extras_require=extras_require,
)
