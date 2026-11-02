cmake_minimum_required(VERSION 3.0.2)
project(deform_monitor_bringup)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/static_planning_evaluation.py
  scripts/planning_core.py
  scripts/mission_map_proxy.py
  scripts/stationary_monitor_cloud_adapter.py
  scripts/lio_twist_odometry.py
  scripts/icra2027_closed_loop_mission.py
  scripts/icra2027_experiment_recorder.py
  scripts/ros_log_scrubber.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/run_icra2027_closed_loop.sh
  scripts/run_icra2027_stationary_full_process.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES scripts/closed_loop_core.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_planning_core.py)
  catkin_add_nosetests(test/test_closed_loop_core.py)
  catkin_add_nosetests(test/test_operator_start.py)
  catkin_add_nosetests(test/test_stationary_monitor_cloud_adapter.py)
  catkin_add_nosetests(test/test_lio_twist_odometry.py)
  catkin_add_nosetests(test/test_experiment_recorder_core.py)
  catkin_add_nosetests(test/test_icra2027_config_isolation.py)
  catkin_add_nosetests(test/test_ros_log_scrubber.py)
  catkin_install_python(PROGRAMS
    test/test_generic_deformation_map.py
    test/test_soft_cost_navigation.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  add_rostest(test/generic_deformation_map.test)
  add_rostest(test/soft_cost_navigation.test)
endif()
