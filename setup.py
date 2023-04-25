from setuptools import setup, find_packages

setup(
    name="Help_protect_great_barrier_reef",
    version="0.1.0",
    packages=find_packages(
        include=["Help_protect_great_barrier_reef", "Help_protect_great_barrier_reef.*"]
    ),
    description="Python programm for the Kaggle competition\
        Help protect the great barrier reef",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Help_Protect_Great_Barrier_Reef",
)