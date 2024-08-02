from setuptools import setup, find_packages

setup(
    name='babylon_sts',
    py_modules=["babylon_sts"],
    version='0.1.40',
    description='A powerful library for audio processing '
                'with advanced features for speech recognition, '
                'text translation, and speech synthesis.',
    author='Artur Rieznik',
    author_email='artuar1990@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'whisper_timestamped',
        'torch',
        'pydub',
        'soundfile',
        'sentencepiece',
        'omegaconf',
        'sacremoses',
        'transformers',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    url="https://github.com/Artuar/babylon_sts",
    include_package_data=True,
)
