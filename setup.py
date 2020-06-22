from setuptools import find_packages, setup

setup(
    name='cleartext',
    version='0.1',
    description='The smart reading assistant for English language learners',
    url='https://bcwallace.com/cleartext/',
    author='Benjamin Wallace',
    author_email='me@bcwallace.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'torch',
        'torchtext',
        'spacy==2.2',
        'click',
        'pytest',
        'flask',
        'mlflow',
    ],
    dependency_links=['https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm'],
)
