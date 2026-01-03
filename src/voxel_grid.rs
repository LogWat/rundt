use std::collections::HashMap; // TODO: hashbrown?
use nalgebra as na;
use rayon::prelude::*;
use crate::point_cloud::{PointCloud, Point, Scalar};

const MIN_POINTS_PER_VOXEL: usize = 6;
const MIN_EIGEN_VALUE: Scalar = 1e-4;

#[derive(Debug, Clone)]
pub struct Voxel {
    pub mean: Point,
    pub cov: na::Matrix3<Scalar>,
    pub inv_cov: na::Matrix3<Scalar>,
    pub num_points: usize,
    pub is_valid: bool,
}

impl Voxel {
    pub fn new() -> Self {
        Self {
            mean: Point::origin(),
            cov: na::Matrix3::zeros(),
            inv_cov: na::Matrix3::zeros(),
            num_points: 0,
            is_valid: false,
        }
    }

    pub fn compute_distribution(&mut self, points: &[Point]) {
        self.num_points = points.len();
        self.is_valid = false;

        if self.num_points < MIN_POINTS_PER_VOXEL {
            return;
        }

        // Compute mean
        let sum = points.iter()
            .fold(na::Vector3::zeros(), |acc, p| acc + p.coords);
        self.mean = Point::from(sum / (self.num_points as Scalar));

        // Compute covariance
        let mut cov = na::Matrix3::zeros();
        for p in points {
            let diff = p.coords - self.mean.coords;
            cov += diff * diff.transpose();
        }
        cov /= self.num_points as Scalar;
        self.cov = cov;

        self.compute_robust_inverse_covariance();
    }

    fn compute_robust_inverse_covariance(&mut self) {
        // Sigma = V * D * V^T
        let eig = self.cov.symmetric_eigen();
        let mut eigen_vals = eig.eigenvalues; // D
        let eigen_vecs = eig.eigenvectors;    // V

        // Clamp eigenvalues
        let mut modified = false;
        for i in 0..3 {
            if eigen_vals[i] < MIN_EIGEN_VALUE {
                eigen_vals[i] = MIN_EIGEN_VALUE;
                modified = true;
            }
        }

        // Sigma' = V * D' * V^T
        if modified {
            let diag = na::Matrix3::from_diagonal(&eigen_vals);
            self.cov = eigen_vecs * diag * eigen_vecs.transpose();
        }

        // Sigma^-1 = V * Lambda^-1 * V^T
        let inv_eigen_vals = na::Vector3::new(
            1.0 / eigen_vals[0],
            1.0 / eigen_vals[1],
            1.0 / eigen_vals[2],
        );
        let inv_diag = na::Matrix3::from_diagonal(&inv_eigen_vals);
        self.inv_cov = eigen_vecs * inv_diag * eigen_vecs.transpose();
        self.is_valid = true;
    }


}

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    resolution: Scalar,
    voxels: HashMap<(i32, i32, i32), Voxel>,
}

impl VoxelGrid {
    pub fn new(resolution: Scalar) -> Self {
        Self {
            resolution,
            voxels: HashMap::new(),
        }
    }

    #[inline]
    fn point_to_key(&self, p: &Point) -> (i32, i32, i32) {
        let ix = (p.x / self.resolution).floor() as i32;
        let iy = (p.y / self.resolution).floor() as i32;
        let iz = (p.z / self.resolution).floor() as i32;
        (ix, iy, iz)
    }

    // PointCloud からボクセルグリッドを生成
    pub fn set_input_cloud(&mut self, cloud: &PointCloud) {
        // grouping
        let mut temp_buckets: HashMap<(i32, i32, i32), Vec<Point>> = HashMap::new();
        for p in &cloud.points {
            let key = self.point_to_key(p);
            temp_buckets
                .entry(key)
                .or_insert_with(Vec::new)
                .push(*p);
        }
        // 正規分布計算
        let new_voxels: HashMap<(i32, i32, i32), Voxel> = temp_buckets
            .into_par_iter()
            .map(|(key, points)| {
                let mut voxel = Voxel::new();
                voxel.compute_distribution(&points);
                (key, voxel)
            })
            .filter(|(_, voxel)| voxel.is_valid)
            .collect();
        self.voxels = new_voxels;
        println!("VoxelGrid: Created {} voxels.", self.voxels.len());
    }

    pub fn get_voxel_at(&self, p: &Point) -> Option<&Voxel> {
        let key = self.point_to_key(p);
        self.voxels.get(&key)
    }

    pub fn get_centroids(&self) -> PointCloud {
        let centroids: Vec<Point> = self.voxels
            .values()
            .map(|voxel| voxel.mean)
            .collect();
        PointCloud::new(centroids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_voxel_distribution_logic() {
        // (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        // mean = (0, 0, 0)
        // covariance = ( (-1-0)^2 + (0-0)^2 + (1-0)^2 ) / 6 = 2/3
        let points = vec![
            Point::new(-1.0, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
            Point::new(-1.0, 0.0, 0.0),
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0),
        ];

        let mut voxel = Voxel::new();
        voxel.compute_distribution(&points);

        assert!(voxel.is_valid);

        assert_relative_eq!(voxel.mean.x, 0.0);
        assert_relative_eq!(voxel.mean.y, 0.0);
        assert_relative_eq!(voxel.mean.z, 0.0);

        // 分散共分散行列の対角成分 [Var(X), Var(Y), Var(Z)]
        let cov = voxel.cov;
        println!("Computed Covariance:\n{}", cov);

        // X軸分散は 2/3
        assert_relative_eq!(cov[(0, 0)], 2.0 / 3.0, epsilon = 1e-6);

        // Y, Z軸分散は 0.0 だが、正則化処理で MIN_EIGEN_VALUE (0.001など) 以上になっているはず
        // 完全に0だと逆行列が作れないため、ここが重要
        assert!(cov[(1, 1)] > 0.0); 
        assert!(cov[(2, 2)] > 0.0);

        // C. 逆行列の検証
        // 逆行列と元の行列を掛けると単位行列になるはず (inv * cov = I)
        let identity_check = voxel.inv_cov * voxel.cov;
        assert_relative_eq!(identity_check[(0, 0)], 1.0, epsilon = 1e-4);
        assert_relative_eq!(identity_check[(1, 1)], 1.0, epsilon = 1e-4);
        assert_relative_eq!(identity_check[(0, 1)], 0.0, epsilon = 1e-4);
    }
}
