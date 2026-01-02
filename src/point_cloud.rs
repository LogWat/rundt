use nalgebra as na;
use rayon::prelude::*;

pub type Scalar = f64;
pub type Point = na::Point3<Scalar>;
pub type Transform = na::Isometry3<Scalar>;

#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point>,
}

impl PointCloud {
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn Transform(&self, transform: &Transform) -> Self {
        let transformed_points: Vec<Point> = self
            .points
            .par_iter()
            .map(|p| transform.transform_point(p))
            .collect();
        Self::new(transformed_points)
    }

    pub fn compute_centroid(&self) -> Option<Point> {
        if self.is_empty() {
            return None;
        }
        let sum = self.points
            .iter() // 並列化のオーバーヘッドのほうが高いかも
            .fold(na::Vector3::zeros(), |acc, p| acc + p.coords);
        Some(Point::from(sum / (self.len() as Scalar)))
    }
}
