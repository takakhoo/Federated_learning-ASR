import setuptools

setuptools.setup(
    name="deepspeech",
    version="0.2.1",
    description="train and evaluate a DeepSpeech or DeepSpeech2 network",
    author="Sam Davis",
    author_email="sam@myrtle.ai",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.5',
    entry_points={
        'console_scripts': ['deepspeech=deepspeech.run:main']
    }
)
