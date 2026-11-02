cmake_minimum_required(VERSION 2.8.3)
project(gflags_catkin)
find_package(catkin_simple REQUIRED)
catkin_simple()
# Use the distribution package instead of downloading at configure time.
find_package(gflags REQUIRED)
cs_install()
cs_export(LIBRARIES gflags)
