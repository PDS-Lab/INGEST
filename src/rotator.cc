#include "rotator.h"

#include <Eigen/Dense>
#include <memory>
#include <myutils/random.h>
#include <myutils/utils.h>
#include <random>

Eigen::MatrixXf gen_random_rotate_matrix(const size_t dim, uint64_t seed) {
  MWC256 rng(seed);
  std::normal_distribution<> dis(0.0, 1.0);
  Eigen::MatrixXd M(dim, dim);
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = 0; j < dim; ++j)
      M(i, j) = dis(rng);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
  Eigen::MatrixXd Q = qr.householderQ();
  if (Q.determinant() < 0)
    Q.col(0) *= -1.0;
  return Q.cast<float>();
}

std::unique_ptr<float[]> init_proj_rand(const size_t dim, const size_t proj_dim,
                                        uint64_t seed) {
  auto res = std::make_unique<float[]>(dim * proj_dim);
  Eigen::MatrixXf Q = gen_random_rotate_matrix(dim, seed);
  Eigen::Map<Eigen::MatrixXf> resA(res.get(), dim, proj_dim);
  resA = (std::sqrt(dim / proj_dim) * Q.leftCols(proj_dim));
  return res;
}

std::unique_ptr<float[]> init_proj_pca(const size_t dim, const size_t proj_dim,
                                       size_t nsample, const float *data,
                                       uint64_t seed) {
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      sampled_edges(data, nsample, dim);
  Eigen::MatrixXf cov =
      (sampled_edges.transpose() * sampled_edges) / (nsample - 1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(cov);
  expect(eigensolver.info() == Eigen::Success);
  auto eigenvalues = eigensolver.eigenvalues().reverse();
  double psum = 0;
  for (size_t i = 0; i < proj_dim; ++i)
    psum += eigenvalues[i];
  double psum_all = psum;
  for (size_t i = proj_dim; i < dim; ++i)
    psum_all += eigenvalues[i];
  double factor = std::sqrt(psum_all / psum);
  auto res = std::make_unique<float[]>(dim * proj_dim);
  Eigen::MatrixXf Q = gen_random_rotate_matrix(proj_dim, seed);
  Eigen::Map<Eigen::MatrixXf> resA(res.get(), dim, proj_dim);
  resA = eigensolver.eigenvectors().rightCols(proj_dim) * Q * factor;
  return res;
}

float *apply_proj(const size_t dim, const size_t proj_dim, const float *input,
                  float *output, float *P) {
  Eigen::Map<const Eigen::RowVectorXf> inputA(input, dim);
  Eigen::Map<Eigen::RowVectorXf> outputA(output, proj_dim);
  Eigen::Map<Eigen::MatrixXf> PA(P, dim, proj_dim);
  outputA = inputA * PA;
  return output;
}

float *apply_proj(const size_t dim, const size_t proj_dim, const int8_t *input,
                  float *output, float *P) {
  Eigen::Map<const Eigen::RowVectorX<int8_t>> inputA(input, dim);
  Eigen::Map<Eigen::RowVectorXf> outputA(output, proj_dim);
  Eigen::Map<Eigen::MatrixXf> PA(P, dim, proj_dim);
  outputA = inputA.cast<float>() * PA;
  return output;
}

float *apply_proj(const size_t dim, const size_t proj_dim, const uint8_t *input,
                  float *output, float *P) {
  Eigen::Map<const Eigen::RowVectorX<uint8_t>> inputA(input, dim);
  Eigen::Map<Eigen::RowVectorXf> outputA(output, proj_dim);
  Eigen::Map<Eigen::MatrixXf> PA(P, dim, proj_dim);
  outputA = inputA.cast<float>() * PA;
  return output;
}
