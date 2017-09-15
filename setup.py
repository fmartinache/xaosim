from setuptools import setup
import os

setup(name='xaosim',
      version='0.2',
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
          'numpy', 'scipy', 'matplotlib'
      ],
      scripts=['bin/zernike_dm', 'bin/shmview'],
      data_files = [(os.getenv('HOME')+'/.config/xaosim/', ['config/shmimview.ui'])],
      zip_safe=False)

'''
execpath = '/usr/local/bin/shmview'
if os.name is 'posix' and os.path.exists(execpath):
    os.chmod(execpath, int('755', 8))
'''
