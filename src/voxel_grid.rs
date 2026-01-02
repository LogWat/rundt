use nalgebra as na;
use crate::point_cloud::{Point, Scalar};

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

    
}