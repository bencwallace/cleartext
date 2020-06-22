from setuptools import setup

setup(
    name='cleartext',
    version='0.1',
    packages=['cleartext'],
    package_dir={'cleartext': 'cleartext'},
    url='',
    license='',
    author='Benjamin Wallace',
    author_email='me@bcwallace.com',
    description='',
    install_requires=[
        'torch',
        'torchtext',
        'spacy',
        'click',
        'pytest',
        'flask',
        'mlflow',
        'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz'
    ]
)
