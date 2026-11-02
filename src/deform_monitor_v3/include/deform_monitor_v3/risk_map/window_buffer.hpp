#ifndef DEFORM_MONITOR_V3_RISK_MAP_WINDOW_BUFFER_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_WINDOW_BUFFER_HPP

#include "deform_monitor_v3/risk_map/risk_types.hpp"

namespace deform_monitor_v3 { namespace risk_map {

class NonOverlappingWindowBuffer {
 public:
  void SetLimits(int frame_count, double max_duration_sec);
  void Reset(uint32_t reference_epoch);
  bool Push(ChannelFrame frame, const FrozenModeVector& block_modes, EvidenceBlock* block);
  bool Flush(EvidenceBlock* block);
  void DiscardIncomplete();
  size_t size() const { return frames_.size(); }

 private:
  bool Emit(bool complete, EvidenceBlock* block);
  int target_frames_ = 5;
  double max_duration_sec_ = 1.0;
  uint32_t epoch_ = 0;
  uint64_t next_block_id_ = 1;
  ChannelFrameDeque frames_;
  FrozenModeVector modes_;
};

} }
#endif
