cmake_minimum_required(VERSION 3.0.2)
project(deform_avoidance_map)

find_package(catkin REQUIRED COMPONENTS
  deform_evolution_map
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  DeformationAvoidanceAlert.msg
  DeformationAvoidanceMap.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    deform_evolution_map
    geometry_msgs
    message_runtime
    nav_msgs
    rospy
    std_msgs
)

catkin_install_python(PROGRAMS
  scripts/deformation_avoidance_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_avoidance_raster.py)
endif()
