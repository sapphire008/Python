from setuptools import setup

setup(
    name='spikedetekt2',
    version='0.3.0',
    author='Klusta-Team',
    author_email='rossant@github',
    packages=[
              'spikedetekt2',
              'spikedetekt2.processing',
              'spikedetekt2.processing.tests',
              'spikedetekt2.core',
              'spikedetekt2.core.tests',
              ],
    entry_points={
          'console_scripts':
              ['spikedetekt = spikedetekt2.core.script:main',
               ]},
    url='http://klusta-team.github.io',
    license='LICENSE.txt',
    description='SpikeDetekt2, part of the KlustaSuite',
    # long_description=open('README.md').read(),
)
