#include "livox_laser_simulation/custommsg_point_writer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

struct Sample {
  ignition::math::Vector3d point;
  double range;
  double intensity;
  uint8_t line;
  uint32_t offset_time;
  int scan_index;
};

livox_ros_driver::CustomMsg BaseMessage() {
  livox_ros_driver::CustomMsg message;
  message.header.seq = 17U;
  message.header.stamp = ros::Time(123U, 456U);
  message.header.frame_id = "livox_frame";
  message.timebase = message.header.stamp.toNSec();
  message.lidar_id = 3U;
  message.rsvd[0] = 7U;
  message.rsvd[1] = 8U;
  message.rsvd[2] = 9U;
  return message;
}

void InitializeVisualizationScan(gazebo::msgs::LaserScan* scan,
                                 int cell_count) {
  ASSERT_NE(scan, nullptr);
  for (int index = 0; index < cell_count; ++index) {
    scan->add_ranges(0.0);
    scan->add_intensities(0.0);
  }
}

void ExpectPointEqual(const livox_ros_driver::CustomPoint& lhs,
                      const livox_ros_driver::CustomPoint& rhs) {
  EXPECT_EQ(lhs.offset_time, rhs.offset_time);
  EXPECT_EQ(lhs.x, rhs.x);
  EXPECT_EQ(lhs.y, rhs.y);
  EXPECT_EQ(lhs.z, rhs.z);
  EXPECT_EQ(lhs.reflectivity, rhs.reflectivity);
  EXPECT_EQ(lhs.tag, rhs.tag);
  EXPECT_EQ(lhs.line, rhs.line);
}

TEST(CustomMsgPointWriter, VisualizationSideChannelIsFieldExact) {
  const std::array<Sample, 4> samples{{
      {ignition::math::Vector3d(1.25, -2.5, 3.75), 4.5, -3.0, 0U, 0U, 0},
      {ignition::math::Vector3d(-0.0, 8.0, -9.0), 12.0, 42.4, 1U, 100U, 1},
      {ignition::math::Vector3d(0.125, 0.25, 0.5), 0.0, 255.0, 2U, 200U, 2},
      {ignition::math::Vector3d(-7.0, 6.0, 5.0), 19.9, 999.0, 3U, 300U, 3},
  }};

  livox_ros_driver::CustomMsg without_visualization = BaseMessage();
  livox_ros_driver::CustomMsg with_visualization = BaseMessage();
  without_visualization.points.reserve(samples.size());
  with_visualization.points.reserve(samples.size());

  gazebo::msgs::LaserScan scan;
  InitializeVisualizationScan(&scan, static_cast<int>(samples.size()));

  for (const Sample& sample : samples) {
    livox_laser_simulation::AppendCustomPointAndOptionalVisualization(
        sample.point, sample.range, sample.intensity, sample.line,
        sample.offset_time, sample.scan_index, nullptr,
        &without_visualization);
    livox_laser_simulation::AppendCustomPointAndOptionalVisualization(
        sample.point, sample.range, sample.intensity, sample.line,
        sample.offset_time, sample.scan_index, &scan, &with_visualization);
  }
  without_visualization.point_num = without_visualization.points.size();
  with_visualization.point_num = with_visualization.points.size();

  EXPECT_EQ(without_visualization.header.seq,
            with_visualization.header.seq);
  EXPECT_EQ(without_visualization.header.stamp,
            with_visualization.header.stamp);
  EXPECT_EQ(without_visualization.header.frame_id,
            with_visualization.header.frame_id);
  EXPECT_EQ(without_visualization.timebase, with_visualization.timebase);
  EXPECT_EQ(without_visualization.lidar_id, with_visualization.lidar_id);
  for (size_t index = 0; index < without_visualization.rsvd.size(); ++index) {
    EXPECT_EQ(without_visualization.rsvd[index],
              with_visualization.rsvd[index]);
  }
  EXPECT_EQ(without_visualization.point_num,
            with_visualization.point_num);
  ASSERT_EQ(without_visualization.points.size(),
            with_visualization.points.size());
  for (size_t index = 0; index < samples.size(); ++index) {
    ExpectPointEqual(without_visualization.points[index],
                     with_visualization.points[index]);
    EXPECT_DOUBLE_EQ(scan.ranges(static_cast<int>(index)),
                     samples[index].range);
    EXPECT_DOUBLE_EQ(scan.intensities(static_cast<int>(index)),
                     samples[index].intensity);
  }
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
