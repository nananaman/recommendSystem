from setuptools import setup, find_packages

setup(name='recommender',
        version='1.0.0',
        description='Recommender',
        author='nananaman',
        packages=find_packages(),
        entry_points="""
        [console_scripts]
        recommender = src.recommender:main
        """,)
