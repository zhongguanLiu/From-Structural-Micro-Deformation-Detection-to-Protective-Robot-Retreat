#!/usr/bin/env python3
"""Read-only saved-world verification before spawning the experiment robot."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def pose_matrix(values):
    x, y, z, roll, pitch, yaw = values
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    matrix = np.eye(4)
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                       [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = [x, y, z]
    return matrix


def saved_scene(content):
    world = ET.fromstring(content).find('world')
    if world is None:
        raise ValueError('World file has no world element')
    result = {}
    for model in world.findall('model'):
        name = model.attrib['name']
        saved = world.find("state/model[@name='{}']".format(name))
        pose = (saved if saved is not None else model).findtext('pose', '0 0 0 0 0 0')
        values = [float(v) for v in pose.split()]
        if len(values) != 6 or not np.isfinite(values).all() or name in result:
            raise ValueError('Invalid/duplicate saved model: '+name)
        controlled = name in {'m01', 'm02', 'm03', 'm04', 'm05', 'm06'}
        static = model.findtext('static', 'false') in ('1', 'true')
        if not static and not controlled:
            raise ValueError('Uncontrolled scene model is not static: '+name)
        links = {}
        for link in model.findall('link'):
            saved_link = saved.find("link[@name='{}']".format(link.attrib['name'])) if saved is not None else None
            if saved_link is not None:
                matrix = pose_matrix([float(v) for v in saved_link.findtext('pose').split()])
            else:
                matrix = pose_matrix(values) @ pose_matrix([
                    float(v) for v in link.findtext('pose', '0 0 0 0 0 0').split()])
            links[link.attrib['name']] = matrix
        result[name] = dict(pose=values, matrix=pose_matrix(values), links=links,
                            static=static, controlled=controlled,
                            pose_source='saved_state' if saved is not None else 'model_pose')
    if not result:
        raise ValueError('No inline scene models to verify')
    return result


def check_pose(expected, pose):
    q = pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    if not np.isfinite([x, y, z, w]).all() or abs(x*x+y*y+z*z+w*w-1) > 1.e-5:
        raise ValueError('Invalid Gazebo orientation')
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    xyz = [pose.position.x, pose.position.y, pose.position.z]
    distance = float(np.linalg.norm(expected[:3, 3]-xyz))
    angle = math.acos(float(np.clip((np.trace(expected[:3, :3].T @ rotation)-1)/2, -1, 1)))
    # Saved SDF decimal precision is ~10 micrometres. Do not tolerate millimetre drift.
    if not math.isfinite(distance) or distance > 1.e-4 or angle > 1.e-4:
        raise ValueError('Saved/runtime pose mismatch: {:.6g} m, {:.6g} rad'.format(distance, angle))
    return dict(position_m=xyz, quaternion_xyzw=[x, y, z, w],
                position_error_m=distance, orientation_error_rad=angle)


def verify_loaded_scene(world_path, output_directory='', motion_config=''):
    import rospy
    from gazebo_msgs.srv import GetModelState, GetModelProperties, GetLinkState, GetLinkProperties
    from std_srvs.srv import Trigger

    path = Path(world_path).resolve(strict=True)
    content = path.read_bytes()
    expected = saved_scene(content)
    # gazebo_ros advertises its services before World::Init applies <state>.
    # World plugins load after Init; this service is the real readiness barrier,
    # including captured reset poses and gravity ownership. No unpause is needed.
    plugin = ET.fromstring(content).find("world/plugin[@name='icra2027_synchronized_motion']")
    if plugin is None:
        raise ValueError('ICRA2027 world is missing its motion plugin')
    ready_service = plugin.findtext('ros_namespace', '/icra2027_motion').rstrip('/')+'/get_status'
    rospy.wait_for_service(ready_service, timeout=30)
    ready = rospy.ServiceProxy(ready_service, Trigger)()
    if not ready.success:
        raise ValueError('Motion plugin initialization failed: '+ready.message)
    plugin_state = json.loads(ready.message)
    if plugin_state.get('physics_ownership_valid') is not True:
        raise ValueError('Motion plugin must verify actual kinematic/gravity state; rebuild the plugin')
    services = {}
    for name, kind in [('get_model_state', GetModelState), ('get_model_properties', GetModelProperties),
                       ('get_link_state', GetLinkState), ('get_link_properties', GetLinkProperties)]:
        rospy.wait_for_service('/gazebo/'+name, timeout=30)
        services[name] = rospy.ServiceProxy('/gazebo/'+name, kind)

    def query(name, *args):
        response = services[name](*args)
        if not response.success:
            raise ValueError('{} {}: {}'.format(name, args, response.status_message))
        return response

    report = dict(status='CHECKING', source_world=str(path),
                  world_sha256=hashlib.sha256(content).hexdigest(),
                  initialization_barrier=ready_service,
                  physics_ownership_valid=True,
                  position_tolerance_m=1.e-4, orientation_tolerance_rad=1.e-4, models={})
    directory = Path(output_directory) if output_directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / 'loaded_scene.world'
        if snapshot.exists() and snapshot.read_bytes() != content:
            raise ValueError('Refusing to mix different worlds in one run directory')
        snapshot.write_bytes(content)
        if motion_config:
            motion = Path(motion_config).resolve(strict=True)
            data = motion.read_bytes()
            (directory / 'motion_config.yaml').write_bytes(data)
            report.update(source_motion_config=str(motion), motion_config_sha256=hashlib.sha256(data).hexdigest())
    try:
        for name, item in expected.items():
            properties = query('get_model_properties', name)
            if properties.is_static != item['static']:
                raise ValueError('Static flag differs from saved world: '+name)
            try:
                observed = check_pose(item['matrix'], query('get_model_state', name, 'world').pose)
            except ValueError as error:
                raise ValueError(name+': '+str(error)) from error
            observed.update(saved_pose=item['pose'], pose_source=item['pose_source'],
                            static=item['static'], controlled=item['controlled'], links={})
            report['models'][name] = observed
            for link, matrix in item['links'].items():
                scoped = name+'::'+link
                observed['links'][link] = check_pose(matrix, query('get_link_state', scoped, 'world').link_state.pose)
                if item['controlled'] and query('get_link_properties', scoped).gravity_mode:
                    raise ValueError('Controlled object still has gravity enabled: '+scoped)
        if path.read_bytes() != content:
            raise ValueError('World file changed during startup verification; restart with a fixed file')
        report.update(status='VERIFIED', verification_sim_time_s=rospy.Time.now().to_sec())
        rospy.logwarn('[closed_loop] SCENE_VERIFIED world=%s sha256=%s models=%d; saved poses preserved',
                      path, report['world_sha256'], len(expected))
    except Exception as error:
        report.update(status='FAILED', error=str(error))
        raise
    finally:
        if directory:
            (directory / 'scene_startup.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
