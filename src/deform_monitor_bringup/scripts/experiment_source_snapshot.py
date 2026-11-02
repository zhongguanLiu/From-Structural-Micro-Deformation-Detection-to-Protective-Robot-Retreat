#!/usr/bin/env python3
"""Freeze bounded, non-secret experiment sources after building, before launch."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PACKAGES = ('deform_monitor_v3', 'deform_evolution_map', 'deform_avoidance_map',
            'deform_monitor_bringup', 'FAST_LIO', 'livox_ros_driver',
            'gazebo_test/Mid360_simulation_plugin/livox_laser_simulation',
            'gazebo_test/Mid360_simulation_plugin/a2_description')
SUFFIXES = {'.py', '.sh', '.cpp', '.cc', '.c', '.h', '.hpp', '.yaml', '.yml',
            '.launch', '.xml', '.urdf', '.xacro', '.msg', '.srv', '.cmake', '.rviz'}
SKIP = {'.git', '__pycache__', 'build', 'devel', 'logs', 'Log', 'PCD',
        'test', 'tests', 'third_party', 'new_disscusion'}
BINARIES = ('deform_monitor_v3/deform_monitor_v1_node', 'FAST_LIO/fastlio_mapping',
            'fast_lio/fastlio_mapping', 'livox_laser_simulation/dual_livox_merge_node',
            'libdeform_monitor_v3.so', 'libicra2027_motion_world_plugin.so',
            'libicra2027_planar_move_plugin.so', 'liblivox_laser_simulation.so',
            'libdual_lidar_trigger_core.so')


def snapshot(workspace, run, launch_arguments):
    target = run / 'provenance'
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / 'execution_sources.zip'
    sources, binaries = {}, {}
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for package in PACKAGES:
            root = workspace / 'src' / package
            if not root.is_dir():
                raise ValueError('Missing source package: '+package)
            for path in sorted(root.rglob('*')):
                relative = path.relative_to(workspace / 'src')
                if path.is_symlink() or not path.is_file() or SKIP.intersection(relative.parts):
                    continue
                if not (path.suffix in SUFFIXES or path.name in ('CMakeLists.txt', 'ICRA2027.world')):
                    continue
                data = path.read_bytes()
                sources[str(relative)] = hashlib.sha256(data).hexdigest()
                archive.writestr(str(relative), data)
    for name in BINARIES:
        path = workspace / 'devel/lib' / name
        if path.is_file():
            binaries[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not binaries:
        raise ValueError('No built experiment binaries found')
    manifest = dict(source_sha256=sources, binary_sha256=binaries,
                    archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    launch_arguments=launch_arguments,
                    scope='Project text sources/configuration and selected built binaries; '
                          'not an archive of system ROS/Gazebo dependencies or mesh assets. '
                          'Loaded world and effective ROS parameters are recorded separately.')
    (target / 'execution_versions.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Execution provenance: {} sources, {} binary hashes'.format(len(sources), len(binaries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('launch_arguments', nargs='*')
    args = parser.parse_args()
    snapshot(args.workspace.resolve(), args.run.resolve(), args.launch_arguments)
