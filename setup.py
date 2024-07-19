from setuptools import setup, find_packages

setup(
    name='babylon_sts',
    version='0.1.0',
    description='A library for audio processing with speech recognition and translation',
    author='Artur Rieznik',
    author_email='artuar1990@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'whisper_timestamped',
        'torch',
        'pydub',
        'transformers',
        'speechrecognition',
        'soundfile'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
