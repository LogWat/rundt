
use nalgebra as na;
use rayon::prelude::*;
use crate::point_cloud::{PointCloud, Point, Scalar};
use crate::voxel_grid::VoxelGrid;

type Vector6 = na::SVector<Scalar, 6>;
type Matrix6 = na::SMatrix<Scalar, 6, 6>;

pub struct NDTMatcher {
    max_iterations: usize,
    epsilon: Scalar,
    step_size: Scalar,
}

impl NDTMatcher {
    pub fn new() -> Self {
        Self {
            max_iterations: 50,
            epsilon: 1e-4,
            step_size: 1.0,
        }
    }

    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.max_iterations = max_iter;
    }
    pub fn set_epsilon(&mut self, epsilon: Scalar) {
        self.epsilon = epsilon;
    }
    pub fn set_step_size(&mut self, step_size: Scalar) {
        self.step_size = step_size;
    }
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }
    pub fn get_epsilon(&self) -> Scalar {
        self.epsilon
    }
    pub fn get_step_size(&self) -> Scalar {
        self.step_size
    }

    pub fn align(
        &self,
        source: &PointCloud,
        target_grid: &VoxelGrid,
        init_guess: na::Isometry3<Scalar>,
        verbose: bool,
    ) -> Option<na::Isometry3<Scalar>> {
        let mut current_transform = init_guess;
        for i in 0..self.max_iterations {
            let (hessian, gradient, score) = source.points.par_iter()
                .map(|point| {
                    let p_trans = current_transform * point;
                    // Find corresponding voxel
                    if let Some(voxel) = target_grid.get_voxel_at(&p_trans) {
                        if !voxel.is_valid { return (Matrix6::zeros(), Vector6::zeros(), 0.0); }
                        // 誤差vec
                        let q = p_trans.coords - voxel.mean.coords;
                        // E = exp(-0.5 * q^T * Sigma^-1 * q)
                        let mahal = (q.transpose() * voxel.inv_cov * q)[0];
                        let score_k = (-0.5 * mahal).max(-50.0).exp(); // avoid underflow
                        // Jacobian
                        let jacobian = Self::compute_point_jacobian(&p_trans);
                        // 勾配 E * (q^T * Sigma^-1 * J)^T
                        let g_k = -(q.transpose() * voxel.inv_cov * jacobian).transpose() * score_k;
                        // hessian (近似) -E * (J^T * Sigma^-1 * J)
                        let h_k = -score_k * (jacobian.transpose() * voxel.inv_cov * jacobian);
                        return (h_k, g_k, score_k)
                    }
                    (Matrix6::zeros(), Vector6::zeros(), 0.0)
                })
                .reduce(
                    || (Matrix6::zeros(), Vector6::zeros(), 0.0),
                    |(h1, g1, s1), (h2, g2, s2)| (h1 + h2, g1 + g2, s1 + s2)
                );
                
            // solve for delta_x: H * delta_x = -g
            let mut h_positive = -hessian;
            let lambda = 0.01; // damping factor (0.1 ~ 10.0)
            for k in 0..6 {
                h_positive[(k, k)] += lambda;
            }

            match h_positive.try_inverse() {
                Some(h_inv) => {
                    let delta = h_inv * gradient;

                    let delta_norm = delta.norm();
                    let final_delta = if delta_norm > 0.5 {
                        delta.normalize() * 0.5
                    } else {
                        delta
                    };

                    Self::update_pose(&mut current_transform, &final_delta, self.step_size);
                    if verbose {
                        println!("Iteration {}: score = {}, delta = {:?}", i, score, final_delta);
                    }

                    if delta_norm < self.epsilon {
                        if verbose {
                            println!("Converged at iter {}", i);
                            println!("Final score: {}", score);
                        }
                        break;
                    }
                },
                None => {
                    eprintln!("Warning: Hessian inversion failed at iter {}. Stopping.", i);
                    break;
                }
            }
        }
        if verbose {
            println!("iterations finished.");
        }
        Some(current_transform)
    }

    // 点pにおける 姿勢 x (6DoF) に対するヤコビ行列 (3x6)
    fn compute_point_jacobian(p: &Point) -> na::SMatrix<Scalar, 3, 6> {
        let mut jacobian = na::SMatrix::<Scalar, 3, 6>::zeros();
        // ∂p/∂tx, ∂p/∂ty, ∂p/∂tz
        jacobian[(0, 0)] = 1.0; jacobian[(1, 1)] = 1.0; jacobian[(2, 2)] = 1.0;
        // ∂p/∂roll, ∂p/∂pitch, ∂p/∂yaw
        let x = p.x; let y = p.y; let z = p.z;
        jacobian[(0, 3)] = 0.0; jacobian[(1, 3)] =  -z; jacobian[(2, 3)] = y;
        jacobian[(0, 4)] = z;   jacobian[(1, 4)] = 0.0; jacobian[(2, 4)] =  -x;
        jacobian[(0, 5)] = -y;  jacobian[(1, 5)] = x;   jacobian[(2, 5)] = 0.0;

        jacobian
    }

    fn update_pose(pose: &mut na::Isometry3<Scalar>, delta: &Vector6, step: Scalar) {
        let translation = na::Translation3::new(
            delta[0] * step, delta[1] * step, delta[2] * step
        );
        let rotation = na::UnitQuaternion::from_euler_angles(
            delta[3] * step, delta[4] * step, delta[5] * step
        );
        let increment = na::Isometry3::from_parts(translation, rotation);
        *pose = increment * (*pose);
    }
}

// ---------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_cloud::PointCloud;
    use crate::voxel_grid::VoxelGrid;
    use nalgebra as na;
    use approx::assert_relative_eq;

    fn create_box_cloud() -> PointCloud {
        let mut points = Vec::new();
        let step = 0.2;
        let mut x = -0.8;
        while x < 0.9 {
            let mut y = -0.4;
            while y < 0.5 {
                let mut z = 0.0;
                while z < 0.9 {
                    points.push(Point::new(x, y, z));
                    z += step;
                }
                y += step;
            }
            x += step;
        }
        PointCloud::new(points)
    }

    #[test]
    fn test_ndt_convergence() {
        let offset = na::Isometry3::translation(1.0, 1.0, 1.0);
        let target_cloud = create_box_cloud().transform(&offset);
        
        let mut voxel_grid = VoxelGrid::new(2.0);
        voxel_grid.set_input_cloud(&target_cloud);

        let source_cloud_origin = target_cloud.clone();
        
        // target: pos (1,1,1), rot (0,0,0)
        // source: pos (1,1,1), rot (0,0,0)
        // initial guess: pos (0.5,0,0), rot (0,0,0.1)
        // expected result: pos (0,0,0), rot (0,0,0)
        let ndt = NDTMatcher::new();
        
        // initial guess
        let noise_translation = na::Translation3::new(0.7, 0.0, 0.0);
        let noise_rotation = na::UnitQuaternion::from_euler_angles(0.0, 0.0, 0.15);
        let init_guess = na::Isometry3::from_parts(noise_translation, noise_rotation);

        // align
        let result_pose = ndt.align(
            &source_cloud_origin, &voxel_grid, init_guess, true
        ).expect("NDT alignment failed");

        // 検証
        let expected_translation = na::Vector3::new(0.0, 0.0, 0.0);
        
        println!("Result Translation: {}", result_pose.translation);
        println!("Result Rotation (Euler): {:?}", result_pose.rotation.euler_angles());

        // 平行移動の誤差
        assert_relative_eq!(
            result_pose.translation.vector.x, expected_translation.x, epsilon = 0.1
        );
        assert_relative_eq!(
            result_pose.translation.vector.y, expected_translation.y, epsilon = 0.1
        );

        // 回転誤差
        let (_r, _p, y) = result_pose.rotation.euler_angles();
        assert_relative_eq!(y, 0.0, epsilon = 0.05);
    }
}
