#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <gazebo/msgs/msgs.hh>
#include <ignition/math/Vector3.hh>
#include <livox_ros_driver/CustomMsg.h>

namespace livox_laser_simulation {

// Append one already validated ray return to the scientific CustomMsg.  The
// optional LaserScan pointer is a visualization side channel only: supplying
// it may update Gazebo's ray display buffer, but cannot alter any CustomPoint
// field.  Keeping both branches in this small pure writer makes the
// visualize=true/false equivalence directly unit-testable.
inline void AppendCustomPointAndOptionalVisualization(
    const ignition::math::Vector3d& point, double range, double intensity,
    uint8_t line, uint32_t offset_time, int scan_index,
    gazebo::msgs::LaserScan* visualization_scan,
    livox_ros_driver::CustomMsg* message) {
  if (!message) return;

  if (visualization_scan) {
    visualization_scan->set_ranges(scan_index, range);
    visualization_scan->set_intensities(scan_index, intensity);
  }

  livox_ros_driver::CustomPoint output;
  output.x = point.X();
  output.y = point.Y();
  output.z = point.Z();
  output.line = line;
  output.tag = 0x10;
  const double bounded_intensity =
      std::max(0.0, std::min(255.0, intensity));
  output.reflectivity =
      static_cast<uint8_t>(std::lround(bounded_intensity));
  output.offset_time = offset_time;
  message->points.push_back(output);
}

}  // namespace livox_laser_simulation
