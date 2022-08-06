from setuptools import setup

setup(
    name='mimicbot_chat',
    version='1.1.0',
    description='Chat utils for mimicbot',
    url='https://github.com/CakeCrusher/mimicbot/tree/master/mimicbot/mimicbot_chat',
    author='Sebastian Sosa',
    author_email='1sebastian1sosa1@gmail.com',
    license='MIT License',
    packages=['mimicbot_chat'],
    install_requires=[
        'pandas',
        'requests',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)