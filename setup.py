from setuptools import setup
import os
from glob import glob

package_name = 'py_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AlbertaBeef',
    maintainer_email='grouby177@gmail.com',
    description='python examples for vision on ROS2 (using rclpy)',
    license='Apache License 2.0',    
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'usbcam_publisher = py_vision.usbcam_publisher:main',        
            'usbcam_subscriber = py_vision.usbcam_subscriber:main',        
            'webinar_demo = py_vision.webinar_demo:main',                
        ],
    },
)
