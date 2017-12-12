#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
        VectorXd rmse(4);
        rmse << 0,0,0,0;

        // check the validay of the following inputs:
        //  * the estimation vector size should not be zero
        //  * the estimation vector size should equal ground truth vector size
        if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
                std::cout << "Invalid estimation or ground_truth data" << std::endl;
                return rmse;
        }

        // accumulate squared residuals
        VectorXd residual(4);
        for (unsigned int i=0; i < estimations.size(); ++i) {
                residual = estimations[i] - ground_truth[i];

                // coefficient-wise multiplication
                residual = residual.array()*residual.array();
                rmse += residual;
        }

        // calculate the mean
        rmse = rmse/estimations.size();

        // calculate the squared root
        rmse = rmse.array().sqrt();

        // return the result
        return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

        // the input is 4 states and the output is 3 radar measurements
        // * range: distance to the object (sqrt(px^2+py^2))
        // * phi = atan(py/px): refererenced counter-clockwise from the x-axis.
        // * range rate: the project of the velocity, v, onto the line.
        MatrixXd Hj = MatrixXd::Zero(3, 4);
        float px = x_state(0);
        float py = x_state(1);
        float vx = x_state(2);
        float vy = x_state(3);

        // pre-compute a set of terms to avoid repeated calculation
        float c1 = px*px + py*py;
        float c2 = sqrt(c1);
        float c3 = (c1*c2);

        // check division by zero
        if (fabs(c1) < 0.0001) {
                std::cout << "CalculateJacobian() - Error - Division by Zero" << std::endl;
                return Hj;
        }

        // compute the Jacobian matrix
        Hj << (px/c2), (py/c2), 0, 0,
                -(py/c1), (px/c1), 0, 0,
                py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

        return Hj;
}
