#include "deform_monitor_v3/v1/rviz_adapter.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace deform_monitor_v3 {
namespace v1 {

TEST(V1RvizAdapter, ScalesOnlyMarkerArrowAndKeepsPhysicalMessageUnchanged) {
  deform_monitor_v3::V1DetectionTile tile;
  tile.common.header.frame_id = "camera_init";
  deform_monitor_v3::V1ReferenceSupportSample support;
  support.position.x = 1.0;
  tile.reference_support.push_back(support);
  deform_monitor_v3::V1SpatialDetectionEvidence detection;
  detection.position = support.position;
  detection.task_scale_probability_mean_identified = true;
  detection.task_scale_probability_mean = 0.75;
  tile.detection_evidence.push_back(detection);
  deform_monitor_v3::V1DeformationSample sample;
  sample.position = support.position;
  sample.displacement_mean.x = 0.001;
  sample.active_probability_mean_identified = true;
  sample.active_probability_mean = 0.75;
  tile.physical_deformation.push_back(sample);

  const auto markers = BuildV1DeformationMarkers(tile, 50.0, 0.02, 1.0);
  ASSERT_EQ(markers.markers.size(), 4U);
  ASSERT_EQ(markers.markers[1].type,
            visualization_msgs::Marker::SPHERE_LIST);
  ASSERT_EQ(markers.markers[1].points.size(), 1U);
  EXPECT_NEAR(markers.markers[1].color.b, 1.0, 1.0e-6);
  ASSERT_EQ(markers.markers[2].type,
            visualization_msgs::Marker::SPHERE_LIST);
  ASSERT_EQ(markers.markers[2].points.size(), 1U);
  ASSERT_EQ(markers.markers[2].colors.size(), 1U);
  EXPECT_NEAR(markers.markers[2].colors[0].r, 0.75, 1.0e-6);
  const auto& arrow = markers.markers.back();
  EXPECT_EQ(arrow.type, visualization_msgs::Marker::LINE_LIST);
  ASSERT_EQ(arrow.points.size(), 6U);
  ASSERT_EQ(arrow.colors.size(), 6U);
  EXPECT_NEAR(arrow.points[1].x - arrow.points[0].x, 0.05, 1.0e-12);
  EXPECT_NEAR(tile.physical_deformation[0].displacement_mean.x,
              0.001, 1.0e-12);
  for (const auto& color : arrow.colors) {
    EXPECT_NEAR(color.r, 0.75, 1.0e-6);
    EXPECT_NEAR(color.g, 0.25, 1.0e-6);
  }
}

TEST(V1RvizAdapter, DefaultDisplayGateShowsOneTenthMillimeter) {
  deform_monitor_v3::V1DetectionTile tile;
  tile.common.header.frame_id = "camera_init";
  deform_monitor_v3::V1DeformationSample below;
  below.displacement_mean.x = 0.00009;
  tile.physical_deformation.push_back(below);
  deform_monitor_v3::V1DeformationSample visible;
  visible.position.y = 1.0;
  visible.displacement_mean.x = 0.00010;
  tile.physical_deformation.push_back(visible);

  const auto markers = BuildV1DeformationMarkers(
      tile, 50.0, 0.005, 1.0);
  // DELETEALL + empty reference/detection lists + one visible arrow.
  ASSERT_EQ(markers.markers.size(), 4U);
  const auto& arrow = markers.markers.back();
  ASSERT_EQ(arrow.points.size(), 6U);
  EXPECT_NEAR(arrow.points[1].x - arrow.points[0].x, 0.005, 1.0e-12);
  EXPECT_DOUBLE_EQ(tile.physical_deformation[0].displacement_mean.x, 0.00009);
  EXPECT_DOUBLE_EQ(tile.physical_deformation[1].displacement_mean.x, 0.00010);
}

TEST(V1RvizAdapter, RealArrowGateUsesLowerBoundAndPreservesRawSamples) {
  deform_monitor_v3::V1DetectionTile tile;
  deform_monitor_v3::V1DeformationSample sample;
  sample.displacement_mean.x = 0.002;
  sample.active_probability_lower = 0.1;
  sample.active_probability_upper = 1.0;
  tile.physical_deformation.push_back(sample);  // uncertainty is not positive evidence
  sample.active_probability_lower = 0.96;
  sample.displacement_mean.x = 0.0005;
  tile.physical_deformation.push_back(sample);  // below physical 1 mm display floor
  sample.displacement_mean.x = 0.002;
  tile.physical_deformation.push_back(sample);  // retained
  sample.active_probability_lower = std::numeric_limits<double>::quiet_NaN();
  tile.physical_deformation.push_back(sample);
  const auto filtered = BuildV1DeformationMarkers(tile, 50.0, 0.05, 1.0, 0.95);
  ASSERT_EQ(filtered.markers.back().points.size(), 6U);
  EXPECT_NEAR(filtered.markers.back().points[1].x, 0.1, 1.0e-12);
  EXPECT_EQ(tile.physical_deformation.size(), 4U);
  EXPECT_DOUBLE_EQ(tile.physical_deformation[0].active_probability_lower, 0.1);
  // Legacy/simulation callers retain all raw display-length-qualified estimates.
  const auto original = BuildV1DeformationMarkers(tile, 50.0, 0.005, 1.0);
  EXPECT_EQ(original.markers.back().points.size(), 24U);
}

TEST(V1RvizAdapter, PacksEveryVisibleVectorWithDirectionAndProbabilityColor) {
  deform_monitor_v3::V1DetectionTile tile;
  tile.common.header.frame_id = "camera_init";

  deform_monitor_v3::V1DeformationSample x_sample;
  x_sample.position.x = 1.0;
  x_sample.displacement_mean.x = 0.001;
  x_sample.active_probability_mean_identified = true;
  x_sample.active_probability_mean = 0.2;
  tile.physical_deformation.push_back(x_sample);

  deform_monitor_v3::V1DeformationSample y_sample;
  y_sample.position.y = 2.0;
  y_sample.displacement_mean.y = -0.002;
  y_sample.active_probability_mean_identified = false;
  y_sample.active_probability_upper = 0.8;
  tile.physical_deformation.push_back(y_sample);

  deform_monitor_v3::V1DeformationSample z_sample;
  z_sample.position.z = 3.0;
  z_sample.displacement_mean.z = 0.003;
  z_sample.active_probability_mean_identified = true;
  z_sample.active_probability_mean = 0.6;
  tile.physical_deformation.push_back(z_sample);

  const auto markers = BuildV1DeformationMarkers(tile, 50.0, 0.005, 1.0);
  // Marker count is independent of the number of physical samples.
  ASSERT_EQ(markers.markers.size(), 4U);
  const auto& arrows = markers.markers.back();
  ASSERT_EQ(arrows.points.size(), 18U);
  ASSERT_EQ(arrows.colors.size(), arrows.points.size());

  // Each six-point block starts with the shaft.  It preserves sample order,
  // source position, direction and display-scaled length.
  EXPECT_DOUBLE_EQ(arrows.points[0].x, 1.0);
  EXPECT_NEAR(arrows.points[1].x - arrows.points[0].x, 0.05, 1.0e-12);
  EXPECT_NEAR(arrows.points[7].y - arrows.points[6].y, -0.10, 1.0e-12);
  EXPECT_NEAR(arrows.points[13].z - arrows.points[12].z, 0.15, 1.0e-12);
  for (size_t index = 0U; index < 6U; ++index) {
    EXPECT_NEAR(arrows.colors[index].r, 0.2, 1.0e-6);
    EXPECT_NEAR(arrows.colors[index + 6U].r, 0.8, 1.0e-6);
    EXPECT_NEAR(arrows.colors[index + 12U].r, 0.6, 1.0e-6);
  }

  // Both head wings terminate behind the tip along the displacement axis;
  // they therefore encode the sign even without an ARROW marker object.
  EXPECT_LT(arrows.points[3].x, arrows.points[2].x);
  EXPECT_LT(arrows.points[5].x, arrows.points[4].x);
  EXPECT_GT(arrows.points[9].y, arrows.points[8].y);
  EXPECT_GT(arrows.points[11].y, arrows.points[10].y);
}

TEST(V1RvizAdapter, FullLodTileStillProducesFourMarkers) {
  deform_monitor_v3::V1DetectionTile tile;
  constexpr size_t kSampleCount = 8000U;
  tile.physical_deformation.reserve(kSampleCount);
  for (size_t index = 0U; index < kSampleCount; ++index) {
    deform_monitor_v3::V1DeformationSample sample;
    sample.position.x = static_cast<double>(index) * 0.01;
    sample.displacement_mean.x = 0.001;
    sample.active_probability_mean_identified = true;
    sample.active_probability_mean =
        static_cast<double>(index % 101U) / 100.0;
    tile.physical_deformation.push_back(sample);
  }

  const auto markers = BuildV1DeformationMarkers(tile, 50.0, 0.005, 1.0);
  ASSERT_EQ(markers.markers.size(), 4U);
  EXPECT_EQ(markers.markers.back().points.size(), 6U * kSampleCount);
  EXPECT_EQ(markers.markers.back().colors.size(), 6U * kSampleCount);
}

TEST(V1RvizAdapter, StatusMakesMonitoringPhaseAndDisplayScaleExplicit) {
  deform_monitor_v3::V1DetectionStatus status;
  status.common.header.frame_id = "camera_init";
  status.phase = 2U;
  status.reference_frozen = true;
  status.gdu_count = 3U;
  geometry_msgs::Point position;
  const auto markers = BuildV1StatusMarkers(
      status, position, 0.3, 50.0,
      "MISSION=MONITORING | EXPERIMENT=12.3s | MOTION=RUNNING 9.3s");
  ASSERT_EQ(markers.markers.size(), 2U);
  EXPECT_NE(markers.markers[1].text.find(
                "MICRO-DEFORMATION MONITORING ACTIVE"),
            std::string::npos);
  EXPECT_NE(markers.markers[1].text.find("displacement x50"),
            std::string::npos);
  EXPECT_NE(markers.markers[1].text.find("EXPERIMENT=12.3s"),
            std::string::npos);
  EXPECT_GT(markers.markers[1].color.g, markers.markers[1].color.r);
}

}  // namespace v1
}  // namespace deform_monitor_v3

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
