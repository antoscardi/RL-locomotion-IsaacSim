from setuptools import find_packages, setup

package_name = 'humanoid_rl_locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mdiego',
    maintainer_email='maintainer@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'walk_rl_network = humanoid_rl_locomotion.walk_rl_network:main'
        ],
    },
)
