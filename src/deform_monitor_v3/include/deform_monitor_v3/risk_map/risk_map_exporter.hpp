#ifndef DEFORM_MONITOR_V3_RISK_MAP_EXPORTER_HPP
#define DEFORM_MONITOR_V3_RISK_MAP_EXPORTER_HPP
#include "deform_monitor_v3/risk_map/risk_types.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
namespace deform_monitor_v3 { namespace risk_map {
class RiskMapExporter {
 public: RiskMapExporter()=default;~RiskMapExporter();bool Start(const RiskMapParams&p,const std::string&world_frame);void Stop();bool Enqueue(const RiskMapFrame&frame);uint64_t dropped()const{return dropped_;}
 private:void Worker();bool Write(const RiskMapFrame&f);RiskMapParams p_;std::deque<RiskMapFrame>q_;std::mutex mutex_;std::condition_variable cv_;std::thread worker_;std::atomic<bool>running_{false};std::atomic<uint64_t>dropped_{0};};
} }
#endif
