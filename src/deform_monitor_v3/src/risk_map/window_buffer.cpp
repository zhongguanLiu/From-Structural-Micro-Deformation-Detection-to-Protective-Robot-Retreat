#include "deform_monitor_v3/risk_map/window_buffer.hpp"

namespace deform_monitor_v3 { namespace risk_map {

void NonOverlappingWindowBuffer::SetLimits(int n, double duration) {
  target_frames_ = std::max(1, n);
  max_duration_sec_ = std::max(0.0, duration);
}

void NonOverlappingWindowBuffer::Reset(uint32_t epoch) {
  frames_.clear(); modes_.clear(); epoch_ = epoch; next_block_id_ = 1;
}

bool NonOverlappingWindowBuffer::Push(ChannelFrame frame,
                                      const FrozenModeVector& block_modes,
                                      EvidenceBlock* block) {
  if (!block) return false;
  if (!frame.channels.empty() && frame.channels.front().key.reference_epoch != epoch_) {
    Reset(frame.channels.front().key.reference_epoch);
  }
  if (frames_.empty()) modes_ = block_modes;
  frames_.push_back(std::move(frame));
  const double span = (frames_.back().stamp - frames_.front().stamp).toSec();
  if (static_cast<int>(frames_.size()) >= target_frames_ ||
      (max_duration_sec_ > 0.0 && span >= max_duration_sec_)) return Emit(true, block);
  return false;
}

bool NonOverlappingWindowBuffer::Flush(EvidenceBlock* block) {
  return Emit(false, block);
}

void NonOverlappingWindowBuffer::DiscardIncomplete() {
  frames_.clear();
  modes_.clear();
}

bool NonOverlappingWindowBuffer::Emit(bool complete, EvidenceBlock* block) {
  if (!block || frames_.empty()) return false;
  block->block_id = next_block_id_++;
  block->reference_epoch = epoch_;
  block->begin_stamp = frames_.front().stamp;
  block->end_stamp = frames_.back().stamp;
  block->frames.swap(frames_);
  block->modes.swap(modes_);
  block->complete = complete;
  return true;
}

} }
