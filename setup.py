from setuptools import setup, find_packages

setup(
    name='TokenSHAP',
    version='0.2.1',
    author='Roni Goldshmidt',
    author_email='roni.goldshmidt@getnexar.com',
    description='Paper Implementation: "TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation"',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ronigold/TokenSHAP',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19',
        'pandas>=1.1',
        'matplotlib>=3.3',
        'scikit-learn>=0.23',
        'transformers>=4.0',
        'tqdm>=4.50',
        'PyYAML>=5.3',
        'requests>=2.24'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    keywords='shapley values interpretation NLP AI',
    project_urls={
        'Bug Reports': 'https://github.com/ronigold/TokenSHAP/issues',
        'Source': 'https://github.com/ronigold/TokenSHAP',
    },
)
