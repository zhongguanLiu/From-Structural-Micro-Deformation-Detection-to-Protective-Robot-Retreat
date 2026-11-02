#include <gtest/gtest.h>

#include "baseline/m3c2/m3c2_baseline.hpp"

namespace {

baseline::PointT MakePoint(float x, float y, float z, float intensity = 0.0f) {
  baseline::PointT point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  return point;
}

baseline::PointCloudPtr MakePlane(float z_offset) {
  baseline::PointCloudPtr cloud(new baseline::PointCloud());
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      cloud->push_back(MakePoint(0.05f * ix, 0.05f * iy, z_offset));
    }
  }
  return cloud;
}

}  // namespace

TEST(M3C2Baseline, ComputesSignedDistanceAndSignificanceOnShiftedPlane) {
  baseline::M3C2Baseline baseline_method;
  baseline::M3C2Config config;
  config.core_voxel_size = 0.05;
  config.normal_radius = 0.12;
  config.cylinder_radius = 0.10;
  config.cylinder_half_length = 0.20;
  config.min_support = 5;
  config.registration_stddev = 0.001;
  config.cluster.tolerance = 0.12;
  config.cluster.min_cluster_size = 2;
  config.cluster.max_cluster_size = 200;
  baseline_method.SetConfig(config);

  baseline::M3C2Result result = baseline_method.Run(MakePlane(0.0f), MakePlane(0.10f));

  ASSERT_FALSE(result.core_records.empty());
  double max_signed_distance = 0.0;
  bool any_significant = false;
  for (const auto& record : result.core_records) {
    max_signed_distance = std::max(max_signed_distance, record.signed_distance);
    any_significant = any_significant || record.significant;
  }
  EXPECT_GT(max_signed_distance, 0.05);
  EXPECT_TRUE(any_significant);
  ASSERT_EQ(result.clusters.size(), 1u);
}

TEST(M3C2Baseline, PreservesSignedMillimeterDistancesAndRegistrationFloor) {
  baseline::M3C2Baseline method;
  baseline::M3C2Config config;
  method.SetConfig(config);
  for (const float shift : {0.f, 0.003f, 0.005f, -0.005f}) {
    auto result = method.Run(MakePlane(0), MakePlane(shift));
    int valid = 0;
    for (const auto& record : result.core_records) {
      if (record.reference_support < 5 || record.query_support < 5) continue;
      ++valid;
      EXPECT_NEAR(record.signed_distance, shift, 1e-7);
      EXPECT_NEAR(record.lod95, 1.96 * config.registration_stddev, 1e-7);
      EXPECT_EQ(record.significant, std::abs(shift) > 1.96 * config.registration_stddev);
    }
    EXPECT_GT(valid, 0);
  }
}

TEST(M3C2Baseline, MissingSupportIsNotAValidZeroDisplacement) {
  baseline::M3C2Baseline method;
  baseline::M3C2Config config;
  method.SetConfig(config);
  const auto result = method.Run(MakePlane(0), MakePlane(1));
  ASSERT_FALSE(result.core_records.empty());
  for (const auto& record : result.core_records) {
    EXPECT_EQ(record.query_support, 0);
    EXPECT_FALSE(record.significant);
  }
}
