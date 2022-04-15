from setuptools import setup, find_packages
from os.path import abspath, join, dirname


# read the contents of your README file
this_directory = abspath(dirname(__file__))
with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='lightweight_ssl',
    version='1.0.0',
    description='An lightweight benchmark for Self-Supervised Learning',
    author='JosefHuang',
    author_email='a3285556aa@gmail.com',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/HuangChiEn/',
    packages=find_packages(exclude=["downstream", "run_script"]),
    keywords=["lightweight framework", "self-supervised learning", "pytorch_lighting"],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
	    'Programming Language :: Python :: 3.9',
    ]
)
