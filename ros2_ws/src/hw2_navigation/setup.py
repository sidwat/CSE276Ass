from glob import glob
from setuptools import setup

setup(name='hw2_navigation', version='0.1.0', packages=['hw2_navigation'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/hw2_navigation']),
                  ('share/hw2_navigation', ['package.xml']),
                  ('share/hw2_navigation/config', glob('config/*')),
                  ('share/hw2_navigation/launch', glob('launch/*.py'))],
      install_requires=['setuptools', 'numpy', 'PyYAML'],
      maintainer='Student', maintainer_email='student@example.com', license='MIT',
      description='HW2 visual feedback navigation starter',
      entry_points={'console_scripts': ['navigator = hw2_navigation.node:main',
                                        'overlay = hw2_navigation.overlay:main',
                                        'simulate = hw2_navigation.simulate:main',
                                        'check_calibration = hw2_navigation.calibration:main']})
