cmake_minimum_required(VERSION 3.0.2)
project(compare_methods_benchmark)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/benchmark_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  find_package(PythonInterp REQUIRED)
  add_test(NAME comparison_contract
    COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test/test_comparison_contract.py)
endif()
