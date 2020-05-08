from setuptools import setup
import os

setup(name='xaosim',
      version='2.0.0', # defined in the __init__ module
      description='Package for eXtreme Adaptive Optics Simulation',
      url='http://github.com/fmartinache/xaosim',
      author='Frantz Martinache',
      author_email='frantz.martinache@oca.eu',
      license='GPL',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 3.6'
      ],
      packages=['xaosim'],
      install_requires=[
          'numpy', 'scipy', 'matplotlib'
      ],
      scripts=['bin/zernike_dm', 'bin/shmview'],
      data_files = [(os.getenv('HOME')+'/.config/xaosim/', ['config/shmimview.ui'])],
      zip_safe=False)
