use hashbrown::HashMap;
use nalgebra as na;
use rayon::prelude::*;
use crate::point_cloud::{PointCloud, Point, Scalar};

const MIN_POINTS_PER_VOXEL: usize = 6;
const MIN_EIGEN_VALUE: Scalar = 1e-4;
pub type VoxelKey = (i32, i32, i32);

#[derive(Debug, Clone)]
pub struct Voxel {
    pub mean: Point,
    pub inv_cov: na::Matrix3<Scalar>,
    pub is_valid: bool,
}

impl Voxel {
    pub fn new() -> Self {
        Self {
            mean: Point::origin(),
            inv_cov: na::Matrix3::zeros(),
            is_valid: false,
        }
    }

    pub fn compute_distribution(&mut self, points: &[Point]) {
        let num_points = points.len();
        self.is_valid = false;

        if num_points < MIN_POINTS_PER_VOXEL {
            return;
        }

        // Compute mean
        let sum = points.iter()
            .fold(na::Vector3::zeros(), |acc, p| acc + p.coords);
        self.mean = Point::from(sum / (num_points as Scalar));

        // Compute covariance
        let mut cov = na::Matrix3::zeros();
        for p in points {
            let diff = p.coords - self.mean.coords;
            cov += diff * diff.transpose();
        }
        cov /= num_points as Scalar;

        self.compute_robust_inverse_covariance(&cov);
    }

    fn compute_robust_inverse_covariance(&mut self, cov: &na::Matrix3<Scalar>) {
        // Sigma = V * D * V^T
        let eig = cov.symmetric_eigen();
        let mut eigen_vals = eig.eigenvalues; // D
        let eigen_vecs = eig.eigenvectors;    // V

        // Clamp eigenvalues
        for i in 0..3 {
            if eigen_vals[i] < MIN_EIGEN_VALUE {
                eigen_vals[i] = MIN_EIGEN_VALUE;
            }
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
    voxels: HashMap<VoxelKey, Voxel>,
}

impl VoxelGrid {
    pub fn new(resolution: Scalar) -> Self {
        Self {
            resolution,
            voxels: HashMap::new(),
        }
    }

    #[inline]
    pub fn point_to_key(&self, p: &Point) -> VoxelKey {
        (
            (p.x / self.resolution).floor() as i32,
            (p.y / self.resolution).floor() as i32,
            (p.z / self.resolution).floor() as i32,
        )
    }

    // PointCloud からボクセルグリッドを生成
    pub fn set_input_cloud(&mut self, cloud: &PointCloud) {
        // grouping
        let mut temp_buckets: HashMap<VoxelKey, Vec<Point>> = HashMap::new();
        for p in &cloud.points {
            let key = self.point_to_key(p);
            temp_buckets
                .entry(key)
                .or_insert_with(Vec::new)
                .push(*p);
        }
        // 正規分布計算
        let new_voxels: HashMap<VoxelKey, Voxel> = temp_buckets
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

    pub fn get_voxel_by_key(&self, key: &VoxelKey) -> Option<&Voxel> {
        self.voxels.get(key)
    }
}

#[derive(Debug, Clone)]
pub struct VoxelGridFilter {
    resolution: Scalar,
    voxel_map: HashMap<VoxelKey, Point>,
}

impl VoxelGridFilter {
    pub fn new(resolution: Scalar) -> Self {
        Self { resolution, voxel_map: HashMap::new() }
    }

    #[inline]
    fn point_to_key(&self, p: &Point) -> VoxelKey {
        (
            (p.x / self.resolution).floor() as i32,
            (p.y / self.resolution).floor() as i32,
            (p.z / self.resolution).floor() as i32,
        )
    }

    pub fn filter(&mut self, cloud: &PointCloud) -> PointCloud {
        self.voxel_map.clear();
        for p in &cloud.points {
            let key = self.point_to_key(p);
            self.voxel_map.entry(key).or_insert(*p);
        }
        let filtered_points: Vec<Point> = self.voxel_map.values().cloned().collect();
        PointCloud::new(filtered_points)
    }
}
