#include "deform_monitor_v3/risk_map/pose_covariance_model.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>

namespace deform_monitor_v3 { namespace risk_map {

PoseCovarianceResult PoseCovarianceModel::Build(const ChannelFrameDeque& frames) const {
  PoseCovarianceResult out;
  const int t_count = static_cast<int>(frames.size());
  if (t_count == 0) return out;
  out.covariance = Eigen::MatrixXd::Zero(6 * t_count, 6 * t_count);
  std::vector<Eigen::Matrix<double, 6, 6>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> factors(t_count);
  for (int t = 0; t < t_count; ++t) {
    Eigen::Matrix<double, 6, 6> sigma =
        0.5 * (frames[t].pose_cov.Sigma_xi + frames[t].pose_cov.Sigma_xi.transpose());
    Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(sigma);
    if (llt.info() != Eigen::Success) {
      sigma += 1e-12 * Eigen::Matrix<double, 6, 6>::Identity();
      llt.compute(sigma);
    }
    if (llt.info() != Eigen::Success) return out;
    factors[t] = llt.matrixL();
  }
  for (int t = 0; t < t_count; ++t) {
    for (int s = 0; s < t_count; ++s) {
      double corr = t == s ? 1.0 : 0.0;
      if (params_.pose_temporal_model == PoseTemporalModel::AR1_POSE) {
        const ros::Time stamp_t = frames[t].pose_cov.stamp.isZero()
                                      ? frames[t].stamp : frames[t].pose_cov.stamp;
        const ros::Time stamp_s = frames[s].pose_cov.stamp.isZero()
                                      ? frames[s].stamp : frames[s].pose_cov.stamp;
        const double dt = std::abs((stamp_t - stamp_s).toSec());
        const double tau = std::max(1e-6, params_.pose_correlation_time_sec);
        corr = std::exp(-dt / tau);
      }
      out.covariance.block<6, 6>(6 * t, 6 * s) = corr * factors[t] * factors[s].transpose();
    }
  }
  out.covariance = 0.5 * (out.covariance + out.covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(out.covariance, Eigen::EigenvaluesOnly);
  if (eig.info() != Eigen::Success) return out;
  out.minimum_eigenvalue = eig.eigenvalues().minCoeff();
  out.valid = out.minimum_eigenvalue >= -1e-9 && out.covariance.allFinite();
  return out;
}

} }
