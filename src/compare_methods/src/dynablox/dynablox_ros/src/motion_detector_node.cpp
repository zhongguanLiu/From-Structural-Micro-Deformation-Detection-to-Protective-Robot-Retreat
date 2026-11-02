#include <gflags/gflags.h>
#include <ros/ros.h>
#include <cstdlib>

#include "dynablox_ros/motion_detector.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_detector");

  // Always add these arguments for proper logging.
  config_utilities::RequiredArguments ra(
      &argc, &argv, {"--logtostderr", "--colorlogtostderr"});

  // Setup logging.
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Setup node.
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  dynablox::MotionDetector motion_detector(nh, nh_private);

  ros::spin();
  // The upstream TsdfServer destructor crashes after ROS has already torn
  // down its global transport state.  This node owns no persistent output;
  // let the OS reclaim its process-local map after the completed run.
  std::_Exit(0);
}
