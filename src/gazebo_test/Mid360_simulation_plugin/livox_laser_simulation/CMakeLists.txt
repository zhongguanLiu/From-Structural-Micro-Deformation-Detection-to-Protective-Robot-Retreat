cmake_minimum_required(VERSION 2.8.3)
project(livox_laser_simulation)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  livox_ros_driver
  message_runtime
  nav_msgs
  roscpp
  rospy
  tf
  sensor_msgs
  std_msgs
  std_srvs
  message_generation
)

find_package(gazebo REQUIRED)
find_package(PCL REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
add_message_files(FILES
    CustomPoint.msg
    CustomMsg.msg
  #   Message2.msg
  )

generate_messages(DEPENDENCIES
  std_msgs
)
#To solve the error which gazebo does not handle the include file well, we must add this line.
include_directories(/usr/include/)

link_directories(${GAZEBO_LIBRARY_DIRS})

#This is designed for whose proto installed in system is not 2.6. We can install the version of proto in local dir

include_directories(
         include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES livox_laser_simulation dual_lidar_trigger_core dual_livox_merge_core synchronized_multi_model_motion_world_plugin icra2027_motion_world_plugin icra2027_planar_move_plugin
        CATKIN_DEPENDS geometry_msgs livox_ros_driver message_runtime nav_msgs roscpp rospy sensor_msgs std_msgs std_srvs tf
)

catkin_install_python(PROGRAMS
  scripts/circle_cmd.py
  scripts/keyboard_cmd.py
  scripts/icra2027_motion_cli.py
  scripts/icra2027_scan_check.py
  scripts/icra2027_scene_check.py
  scripts/icra2027_truth_recorder.py
  scripts/model_motion_controller.py
  scripts/sim_clock_joint_state_publisher.py
  scripts/spawn_or_replace_model.py
  scripts/synchronized_motion_cli.py
  scripts/unpause_synchronized_dual_lidar.py
  scripts/work_patrol_cmd.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(dual_lidar_trigger_core SHARED src/dual_lidar_trigger.cpp)
target_link_libraries(dual_lidar_trigger_core ${catkin_LIBRARIES})

add_library(livox_laser_simulation SHARED src/livox_points_plugin.cpp src/livox_ode_multiray_shape.cpp)
add_dependencies(livox_laser_simulation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(livox_laser_simulation dual_lidar_trigger_core ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${PCL_LIBRARIES} RayPlugin)

add_library(synchronized_multi_model_motion_world_plugin SHARED
  src/synchronized_multi_model_motion_world_plugin.cpp)
add_dependencies(synchronized_multi_model_motion_world_plugin
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(synchronized_multi_model_motion_world_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(icra2027_motion_world_plugin SHARED
  src/icra2027_motion_world_plugin.cpp)
add_dependencies(icra2027_motion_world_plugin
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(icra2027_motion_world_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_library(icra2027_planar_move_plugin SHARED
  src/icra2027_planar_move_plugin.cpp)
add_dependencies(icra2027_planar_move_plugin
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(icra2027_planar_move_plugin
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_executable(livox_custommsg_to_pointcloud2_node
  src/livox_custommsg_to_pointcloud2_node.cpp)
add_dependencies(livox_custommsg_to_pointcloud2_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(livox_custommsg_to_pointcloud2_node ${catkin_LIBRARIES})

add_library(dual_livox_merge_core src/dual_livox_merge.cpp)
add_dependencies(dual_livox_merge_core
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dual_livox_merge_core ${catkin_LIBRARIES})

add_executable(dual_livox_merge_node src/dual_livox_merge_node.cpp)
add_dependencies(dual_livox_merge_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dual_livox_merge_node
  dual_livox_merge_core ${catkin_LIBRARIES})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(test_kinematic_imu test/test_kinematic_imu.cpp)
  if(TARGET test_kinematic_imu)
    target_link_libraries(test_kinematic_imu ${GAZEBO_LIBRARIES})
  endif()
  catkin_add_nosetests(test/test_icra2027_contract.py)
  catkin_add_nosetests(test/test_work_patrol_launch.py)
  catkin_add_gtest(test_dual_livox_merge test/test_dual_livox_merge.cpp)
  if(TARGET test_dual_livox_merge)
    target_link_libraries(test_dual_livox_merge
      dual_livox_merge_core ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_dual_lidar_trigger test/test_dual_lidar_trigger.cpp)
  if(TARGET test_dual_lidar_trigger)
    target_link_libraries(test_dual_lidar_trigger
      dual_lidar_trigger_core ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_custommsg_point_writer
    test/test_custommsg_point_writer.cpp)
  if(TARGET test_custommsg_point_writer)
    add_dependencies(test_custommsg_point_writer
      ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
    target_link_libraries(test_custommsg_point_writer
      ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
  endif()
endif()

# Standalone, opt-in relay used only by the dedicated disturbance experiment.
# Normal Gazebo and FAST-LIO launches do not start or link against this node.
add_executable(livox_custommsg_disturbance_node
  src/livox_custommsg_disturbance_node.cpp)
add_dependencies(livox_custommsg_disturbance_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(livox_custommsg_disturbance_node ${catkin_LIBRARIES})
