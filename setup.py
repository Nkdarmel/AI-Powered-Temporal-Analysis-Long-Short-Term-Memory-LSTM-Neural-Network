#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for AI-Powered Temporal Analysis using LSTM Neural Networks
"""

from setuptools import setup, find_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the contents of README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lstm-cybersecurity',
    version='0.1.0',
    description='AI-Powered Temporal Analysis for Cybersecurity using LSTM Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nkdarmel',
    author_email='nkdarmel@example.com',
    url='https://github.com/Nkdarmel/AI-Powered-Temporal-Analysis-Long-Short-Term-Memory-LSTM-Neural-Network',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.9.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
            'pylint>=2.18.0',
            'mypy>=1.5.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'deployment': [
            'fastapi>=0.103.0',
            'uvicorn>=0.23.0',
            'docker>=6.1.0',
        ],
        'monitoring': [
            'wandb>=0.15.0',
            'neptune-client>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
        'Topic :: System :: Monitoring',
    ],
    keywords='lstm cybersecurity anomaly-detection network-traffic temporal-analysis deep-learning',
    project_urls={
        'Documentation': 'https://github.com/Nkdarmel/AI-Powered-Temporal-Analysis-Long-Short-Term-Memory-LSTM-Neural-Network/tree/main/docs',
        'Source': 'https://github.com/Nkdarmel/AI-Powered-Temporal-Analysis-Long-Short-Term-Memory-LSTM-Neural-Network',
        'Issues': 'https://github.com/Nkdarmel/AI-Powered-Temporal-Analysis-Long-Short-Term-Memory-LSTM-Neural-Network/issues',
    },
    python_requires='>=3.8,<3.12',
    zip_safe=False,
)