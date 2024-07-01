from setuptools import find_packages, setup

package_name = 'adonis_mot'

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
    maintainer='colino',
    maintainer_email='francisco.m.colino@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cluster_bbox_viz = adonis_mot.cluster_bbox_viz:main',
            'cluster_bbox_pc_fusion_viz = adonis_mot.cluster_bbox_pc_fusion_viz:main',
            'ocsort_cluster_bbox_viz = adonis_mot.ocsort_cluster_bbox_viz:main',
            'ocsort_cluster_bbox_label_viz = adonis_mot.ocsort_cluster_bbox_label_viz:main',
            'occupancy_grid_3d_viz = adonis_mot.occupancy_grid_3d_viz:main',
            'cluster_tracking_occupancy_pub = adonis_mot.cluster_tracking_occupancy_pub:main',
            'bbox_tracking_occupancy_pub = adonis_mot.bbox_tracking_occupancy_pub:main',
            'db_bbox_tracking_occ_viz = adonis_mot.db_bbox_tracking_occ_viz:main',
        ],
    },
)
