#include "livox_laser_simulation/dual_lidar_trigger.hpp"

#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace livox_laser_simulation {
namespace {

TEST(DualLidarTrigger, ParsesOnlyExplicitRoles) {
  DualLidarTriggerRole role = DualLidarTriggerRole::PRIMARY;
  EXPECT_TRUE(ParseDualLidarTriggerRole("primary", &role));
  EXPECT_EQ(role, DualLidarTriggerRole::PRIMARY);
  EXPECT_TRUE(ParseDualLidarTriggerRole("secondary", &role));
  EXPECT_EQ(role, DualLidarTriggerRole::SECONDARY);
  EXPECT_FALSE(ParseDualLidarTriggerRole("rear", &role));
  EXPECT_FALSE(ParseDualLidarTriggerRole("", &role));
  EXPECT_FALSE(ParseDualLidarTriggerRole("primary", nullptr));
}

TEST(DualLidarTrigger, DeliversExactPrimaryAcquisitionTimestampOnce) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  uint64_t received_stamp = 0U;
  uint64_t callback_count = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [&](uint64_t stamp) {
        received_stamp = stamp;
        ++callback_count;
        return true;
      },
      &secondary_token, &error));
  EXPECT_TRUE(registry.GroupReady("pair"));
  EXPECT_TRUE(registry.TriggerPair(
      primary_token, 1234567890123456ULL, &error));
  EXPECT_EQ(received_stamp, 1234567890123456ULL);
  EXPECT_EQ(callback_count, 1U);
}

TEST(DualLidarTrigger,
     SavedWorldCatchUpIsSuppressedBeforePublishingWithoutChangingEpochs) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  uint64_t suppressed_count = 0U;
  std::vector<uint64_t> primary_epochs;
  std::vector<uint64_t> secondary_epochs;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "saved_world_pair", DualLidarTriggerRole::PRIMARY,
      [&](uint64_t epoch) {
        primary_epochs.push_back(epoch);
        return true;
      },
      &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "saved_world_pair", DualLidarTriggerRole::SECONDARY,
      [&](uint64_t epoch) {
        secondary_epochs.push_back(epoch);
        return true;
      },
      &secondary_token, &error));

  constexpr uint64_t kSavedWorldEpochNs = 1103000000000ULL;
  constexpr uint64_t kConfiguredTenHertzPeriodNs = 100000000ULL;
  EXPECT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token, kSavedWorldEpochNs,
                kConfiguredTenHertzPeriodNs, &suppressed_count, &error),
            DualLidarTriggerResult::TRIGGERED);
  EXPECT_EQ(suppressed_count, 0U);

  // This is the 18 ms startup burst seen after loading the saved ICRA world.
  // It is a scheduler callback, not a measurement and reaches neither source
  // publisher nor the merger.
  EXPECT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token, kSavedWorldEpochNs + 18000000ULL,
                kConfiguredTenHertzPeriodNs, &suppressed_count, &error),
            DualLidarTriggerResult::SCHEDULER_CATCH_UP_SUPPRESSED);
  EXPECT_EQ(suppressed_count, 1U);
  EXPECT_NE(error.find("scheduler catch-up"), std::string::npos);

  // Suppression does not advance the accepted anchor.  The exact configured
  // boundary remains a legal second physical acquisition with its original
  // Gazebo timestamp.
  EXPECT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token,
                kSavedWorldEpochNs + kConfiguredTenHertzPeriodNs,
                kConfiguredTenHertzPeriodNs, &suppressed_count, &error),
            DualLidarTriggerResult::TRIGGERED);
  EXPECT_EQ(suppressed_count, 1U);
  const std::vector<uint64_t> expected = {
      kSavedWorldEpochNs,
      kSavedWorldEpochNs + kConfiguredTenHertzPeriodNs};
  EXPECT_EQ(primary_epochs, expected);
  EXPECT_EQ(secondary_epochs, expected);
}

TEST(DualLidarTrigger, CatchUpGateStillRejectsDuplicateOrReversedEpochs) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  uint64_t suppressed_count = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [](uint64_t) { return true; }, &secondary_token, &error));
  ASSERT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token, 200U, 100U, &suppressed_count, &error),
            DualLidarTriggerResult::TRIGGERED);
  EXPECT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token, 200U, 100U, &suppressed_count, &error),
            DualLidarTriggerResult::ERROR);
  EXPECT_EQ(registry.TriggerPairWithMinimumInterval(
                primary_token, 199U, 100U, &suppressed_count, &error),
            DualLidarTriggerResult::ERROR);
  EXPECT_EQ(suppressed_count, 0U);
}

TEST(DualLidarTrigger, IncompleteGroupIsGatedBeforeAnyAcquisition) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  uint64_t secondary_acquisitions = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));

  // This is the plugin's pre-ray-cast gate: an initialization callback while
  // only the primary exists must not become a primary-only ROS message.
  bool primary_generated = false;
  if (registry.GroupReady("pair")) primary_generated = true;
  EXPECT_FALSE(primary_generated);

  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [&](uint64_t) {
        ++secondary_acquisitions;
        return true;
      },
      &secondary_token, &error));
  if (registry.GroupReady("pair")) {
    primary_generated = true;
    ASSERT_TRUE(registry.TriggerPair(primary_token, 100U, &error));
  }
  EXPECT_TRUE(primary_generated);
  EXPECT_EQ(secondary_acquisitions, 1U);
}

TEST(DualLidarTrigger, OverlappingCallbacksPublishWholePairsAtomically) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  std::atomic<bool> first_primary_entered{false};
  std::mutex events_mutex;
  std::vector<std::string> events;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [&](uint64_t stamp) {
        {
          std::lock_guard<std::mutex> lock(events_mutex);
          events.push_back("P" + std::to_string(stamp));
        }
        if (stamp == 100U) {
          first_primary_entered.store(true);
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return true;
      },
      &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [&](uint64_t stamp) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back("S" + std::to_string(stamp));
        return true;
      },
      &secondary_token, &error));

  bool first_ok = false;
  bool second_ok = false;
  std::thread first([&]() {
    std::string local_error;
    first_ok = registry.TriggerPair(primary_token, 100U, &local_error);
  });
  while (!first_primary_entered.load()) std::this_thread::yield();
  std::thread second([&]() {
    std::string local_error;
    second_ok = registry.TriggerPair(primary_token, 200U, &local_error);
  });
  first.join();
  second.join();

  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  const std::vector<std::string> expected = {
      "P100", "S100", "P200", "S200"};
  EXPECT_EQ(events, expected);
}

TEST(DualLidarTrigger, RejectsDuplicateAndOutOfOrderEpochBeforePublishing) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  uint64_t primary_count = 0U;
  uint64_t secondary_count = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [&](uint64_t) {
        ++primary_count;
        return true;
      },
      &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [&](uint64_t) {
        ++secondary_count;
        return true;
      },
      &secondary_token, &error));
  ASSERT_TRUE(registry.TriggerPair(primary_token, 200U, &error));
  EXPECT_FALSE(registry.TriggerPair(primary_token, 200U, &error));
  EXPECT_FALSE(registry.TriggerPair(primary_token, 100U, &error));
  EXPECT_EQ(primary_count, 1U);
  EXPECT_EQ(secondary_count, 1U);
  EXPECT_TRUE(registry.TriggerPair(primary_token, 201U, &error));
  EXPECT_EQ(primary_count, 2U);
  EXPECT_EQ(secondary_count, 2U);
}

TEST(DualLidarTrigger, RejectsIncompleteDuplicateAndWrongRoleTriggers) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t duplicate_token = 0U;
  uint64_t secondary_token = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));
  EXPECT_FALSE(registry.GroupReady("pair"));
  EXPECT_FALSE(registry.TriggerPair(primary_token, 1U, &error));
  EXPECT_FALSE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &duplicate_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [](uint64_t) { return true; }, &secondary_token, &error));
  EXPECT_FALSE(registry.TriggerPair(secondary_token, 1U, &error));
}

TEST(DualLidarTrigger, UnregisterRemovesReadinessAndPreventsUseAfterLifetime) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [](uint64_t) { return true; }, &secondary_token, &error));
  registry.Unregister(secondary_token);
  EXPECT_FALSE(registry.GroupReady("pair"));
  EXPECT_FALSE(registry.TriggerPair(primary_token, 1U, &error));
  registry.Unregister(primary_token);
  registry.Unregister(primary_token);
}

TEST(DualLidarTrigger, PropagatesSecondaryAcquisitionFailure) {
  DualLidarTriggerRegistry registry;
  uint64_t primary_token = 0U;
  uint64_t secondary_token = 0U;
  std::string error;
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::PRIMARY,
      [](uint64_t) { return true; }, &primary_token, &error));
  ASSERT_TRUE(registry.Register(
      "pair", DualLidarTriggerRole::SECONDARY,
      [](uint64_t) { return false; }, &secondary_token, &error));
  EXPECT_FALSE(registry.TriggerPair(primary_token, 5U, &error));
  EXPECT_NE(error.find("rejected"), std::string::npos);
}

}  // namespace
}  // namespace livox_laser_simulation

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
