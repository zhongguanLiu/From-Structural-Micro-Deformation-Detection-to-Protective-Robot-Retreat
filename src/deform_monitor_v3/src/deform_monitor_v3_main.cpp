#include "deform_monitor_v3/deform_monitor_v3_node.hpp"

#include <iostream>

int main(int argc, char** argv) {
  ros::init(argc, argv, "deform_monitor_v3_node");

  try {
    deform_monitor_v3::DeformMonitorV3Node node;
    node.Run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[微动监测V3] 异常退出: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[微动监测V3] 未知异常退出" << std::endl;
  }
  return 1;
}
