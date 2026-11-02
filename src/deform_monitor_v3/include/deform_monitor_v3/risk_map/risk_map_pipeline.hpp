#ifndef DEFORM_MONITOR_V3_RISK_MAP_RISK_MAP_PIPELINE_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_RISK_MAP_PIPELINE_HPP
#include "deform_monitor_v3/risk_map/belief_map.hpp"
#include "deform_monitor_v3/risk_map/channel_adapter.hpp"
#include "deform_monitor_v3/risk_map/deformation_evidence.hpp"
#include "deform_monitor_v3/risk_map/emergence_evidence.hpp"
#include "deform_monitor_v3/risk_map/exclusive_association.hpp"
#include "deform_monitor_v3/risk_map/hazard_projector.hpp"
#include "deform_monitor_v3/risk_map/match_evidence.hpp"
#include "deform_monitor_v3/risk_map/visibility_model.hpp"
#include "deform_monitor_v3/risk_map/window_buffer.hpp"
namespace deform_monitor_v3 { namespace risk_map {
class RiskMapPipeline {
 public:
  void SetParams(const RiskMapParams& p);
  void Reset(uint32_t epoch);
  void BuildReference(const ObservationFrameDeque& initialization_frames,uint32_t epoch);
  bool Process(const ObservationFrameDeque& frames,
               const std::vector<CurrentObservationVector>& per_frame_observations,
               const AnchorReferenceVector& anchors,const AnchorStateVector& previous_states,
               uint32_t epoch,RiskMapFrame* output);
  const ReferenceFreeSpaceMap& reference_map()const{return reference_;}
 private:
  void ExtractRegions(RiskMapFrame* frame)const;
  RiskMapParams p_; uint64_t frame_index_=0,event_id_=1; uint32_t epoch_=0; ros::Time last_input_stamp_;
  FrozenModeVector active_block_modes_;
  ChannelAdapter adapter_; NonOverlappingWindowBuffer window_; DeformationEvidence deformation_;
  ReferenceFreeSpaceMap reference_; CurrentOccupancyFrame current_; VisibilityModel visibility_;
  MatchEvidence match_; EmergenceEvidence emergence_; ExclusiveAssociation association_;
  BeliefMap beliefs_; HazardProjector projector_;
};
} }
#endif
