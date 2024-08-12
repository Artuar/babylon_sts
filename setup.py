from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='babylon_sts',
    py_modules=["babylon_sts"],
    version='0.1.41',
    description='A powerful library for audio processing '
                'with advanced features for speech recognition, '
                'text translation, and speech synthesis.',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
