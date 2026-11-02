#include "deform_monitor_v3/v1/node.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "deform_monitor_v1");
  deform_monitor_v3::v1::V1Node node;
  if (!node.Initialize()) return 2;
  node.Spin();
  return 0;
}
