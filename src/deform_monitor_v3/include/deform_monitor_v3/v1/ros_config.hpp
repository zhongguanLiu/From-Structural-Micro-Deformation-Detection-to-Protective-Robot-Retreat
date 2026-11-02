#ifndef DEFORM_MONITOR_V3_V1_ROS_CONFIG_HPP
#define DEFORM_MONITOR_V3_V1_ROS_CONFIG_HPP

#include "deform_monitor_v3/v1/types.hpp"

#include <ros/node_handle.h>

namespace deform_monitor_v3 {
namespace v1 {

bool LoadStrictRosConfig(const ros::NodeHandle& private_node, V1Config* config,
                         RobotGeometry* robot, std::vector<std::string>* errors);

}  // namespace v1
}  // namespace deform_monitor_v3

#endif  // DEFORM_MONITOR_V3_V1_ROS_CONFIG_HPP
