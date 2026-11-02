#include "deform_monitor_v3/risk_map/deformation_evidence.hpp"

#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>

namespace deform_monitor_v3 { namespace risk_map {

EvidenceResult DeformationEvidence::Evaluate(const EvidenceBlock& block, int anchor_id,
                                             const Eigen::MatrixXd& shared) const {
  EvidenceResult out;
  out.covariance_model = params_.pose_temporal_model;
  if (!block.complete) { out.status = EvidenceStatus::INCOMPLETE_BLOCK; return out; }
  const PoseCovarianceResult pose = pose_model_.Build(block.frames);
  if (!pose.valid) { out.status = EvidenceStatus::INVALID_COVARIANCE; return out; }
  std::vector<const ChannelObservation*> rows;
  std::vector<int> frame_indices;
  for (size_t t = 0; t < block.frames.size(); ++t) {
    for (const auto& row : block.frames[t].channels) {
      if (row.key.anchor_id == anchor_id) { rows.push_back(&row); frame_indices.push_back(t); }
    }
  }
  const int n = static_cast<int>(rows.size());
  if (n == 0) { out.status = EvidenceStatus::EMPTY; return out; }
  Eigen::VectorXd z(n), psi(n);
  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n, pose.covariance.rows());
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(rows[i]->independent_variance) ||
        rows[i]->independent_variance <= 0.0) {
      out.status = EvidenceStatus::INVALID_COVARIANCE; return out;
    }
    z[i] = rows[i]->z; psi[i] = rows[i]->psi;
    sigma(i, i) = std::max(params_.independent_variance_floor,
                           rows[i]->independent_variance);
    a.block<1, 6>(i, 6 * frame_indices[i]) = rows[i]->g.transpose();
  }
  sigma.noalias() += a * pose.covariance * a.transpose();
  std::array<Eigen::Matrix3d,256> reference_covariances;
  std::array<bool,256> reference_group_seen{};
  for(const auto*row:rows){const uint8_t group=row->reference_covariance_group;if(group==255)continue;
    const Eigen::Matrix3d covariance=.5*(row->reference_covariance_R+row->reference_covariance_R.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(covariance,Eigen::EigenvaluesOnly);
    if(!covariance.allFinite()||eig.info()!=Eigen::Success||eig.eigenvalues().minCoeff()<-1.0e-12||
       (reference_group_seen[group]&&!reference_covariances[group].isApprox(covariance,1.0e-9))){
      out.status=EvidenceStatus::INVALID_COVARIANCE;return out;}
    reference_covariances[group]=covariance;reference_group_seen[group]=true;}
  for(int i=0;i<n;++i)for(int j=0;j<n;++j){const uint8_t group=rows[i]->reference_covariance_group;
    if(group==255||group!=rows[j]->reference_covariance_group)continue;
    sigma(i,j)+=rows[i]->h_R.dot(reference_covariances[group]*rows[j]->h_R);}
  if (shared.size()!=0) {
    if(shared.rows()!=n||shared.cols()!=n||!shared.allFinite()){
      out.status=EvidenceStatus::INVALID_COVARIANCE;return out;
    }
    const Eigen::MatrixXd symmetric_shared=.5*(shared+shared.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> shared_eig(
        symmetric_shared,Eigen::EigenvaluesOnly);
    if(!shared.isApprox(symmetric_shared,1.0e-9)||shared_eig.info()!=Eigen::Success||
       shared_eig.eigenvalues().minCoeff()<-1.0e-12){
      out.status=EvidenceStatus::INVALID_COVARIANCE;return out;
    }
    sigma+=symmetric_shared;
  }
  sigma = 0.5 * (sigma + sigma.transpose());
  Eigen::LLT<Eigen::MatrixXd> llt;
  double jitter = 0.0;
  for (int attempt = 0; attempt <= params_.covariance_jitter_attempts; ++attempt) {
    Eigen::MatrixXd candidate = sigma;
    if (jitter > 0.0) candidate.diagonal().array() += jitter;
    llt.compute(candidate);
    out.factorization_attempts = attempt + 1;
    if (llt.info() == Eigen::Success) { sigma.swap(candidate); break; }
    jitter = attempt == 0 ? params_.covariance_jitter_initial : jitter * 10.0;
  }
  out.jitter_used = jitter;
  if (llt.info() != Eigen::Success) { out.status = EvidenceStatus::FACTORIZATION_FAILED; return out; }
  const Eigen::VectorXd v = llt.solve(psi);
  const Eigen::VectorXd w = llt.solve(z);
  out.solve_residual = std::max((sigma * v - psi).norm() / std::max(1.0, psi.norm()),
                                (sigma * w - z).norm() / std::max(1.0, z.norm()));
  out.kappa = psi.dot(v);
  if (!std::isfinite(out.kappa) || out.kappa <= params_.kappa_floor) {
    out.status = EvidenceStatus::UNOBSERVABLE_MODE; return out;
  }
  const double varpi = psi.dot(w);
  const double sigma2 = params_.sigma_mu * params_.sigma_mu;
  out.gamma = sigma2 * out.kappa;
  out.xi = varpi / std::sqrt(out.kappa);
  out.log_bayes_factor = 0.5 * (sigma2 * varpi * varpi / (1.0 + out.gamma) -
                                      std::log1p(out.gamma));
  const Eigen::VectorXd diagonal = llt.matrixL().toDenseMatrix().diagonal();
  const double min_diag = diagonal.cwiseAbs().minCoeff();
  const double max_diag = diagonal.cwiseAbs().maxCoeff();
  out.condition_estimate = min_diag > 0.0 ? (max_diag / min_diag) * (max_diag / min_diag)
                                          : std::numeric_limits<double>::infinity();
  out.channel_count = n;
  out.valid = std::isfinite(out.log_bayes_factor) && out.solve_residual < 1e-5;
  out.status = out.valid ? EvidenceStatus::OK : EvidenceStatus::INVALID_COVARIANCE;
  return out;
}

} }
