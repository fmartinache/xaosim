from setuptools import setup
import os

setup(name='xaosim',
      version='0.1',
      description='Package for eXtreme Adaptive Optics Simulation',
      url='http://github.com/fmartinache/xaosim',
      author='Frantz Martinache',
      author_email='frantz.martinache@oca.eu',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 2.7'
      ],
      packages=['xaosim'],
      install_requires=[
          'numpy', 'scipy', 'matplotlib', 'pygame'
      ],
      scripts=['bin/shmview', 'bin/zernike_dm'],
      zip_safe=False)


execpath = '/usr/local/bin/shmview'
if os.name is 'posix' and os.path.exists(execpath):
    os.chmod(execpath, int('755', 8))
