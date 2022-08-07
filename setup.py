from setuptools import setup

setup(
    name='mimicbot_cli',
    version='1.0.0',
    description='Tools and a pipeline for creating of an AI chat bot modeled to imitate a user',
    url='https://github.com/CakeCrusher/mimicbot',
    author='Sebastian Sosa',
    author_email='1sebastian1sosa1@gmail.com',
    license='MIT License',
    packages=['mimicbot_cli'],
    install_requires=[
        'configparser',
        'typer',
        'huggingface-hub',
        'pandas',
        'numpy',
        'torch',
        'torchvision',
        'torchaudio',
        'requests',
        'discord.py',
        'transformers',
        'datasets',
        'rouge-score',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
