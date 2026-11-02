cmake_minimum_required(VERSION 3.0.2)
project(deform_evolution_map)

find_package(catkin REQUIRED COMPONENTS
  deform_monitor_v3
  geometry_msgs
  message_generation
  rospy
  std_msgs
  visualization_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  GduForecast.msg
  GduEvolution.msg
  DeformationEvolutionFrame.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    deform_monitor_v3
    geometry_msgs
    message_runtime
    rospy
    std_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/deformation_evolution_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_evolution_model.py)
  catkin_add_nosetests(test/test_evolution_node_contract.py)
endif()
