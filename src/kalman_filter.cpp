#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd& x_in,
                        MatrixXd& P_in,
                        MatrixXd& F_in,
                        MatrixXd& Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::MeasurementUpdate(const VectorXd& y,
                                     const MatrixXd& H,
                                     const MatrixXd& R) {
  // VectorXd y = z - z_pred;
  MatrixXd temp = P_ * H.transpose();
  MatrixXd S = H * temp + R;
  MatrixXd K = temp * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I  = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}
