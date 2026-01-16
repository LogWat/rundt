use anyhow::{Context, Result};
use clap::Parser;
use pcd_rs::{DynRecord, Reader};
use std::path::PathBuf;
use nalgebra as na;

use rsndt::{
    point_cloud::{Point, PointCloud}, voxel_grid, ndt::NDTMatcher
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    target: PathBuf,

    #[arg(short, long)]
    source: PathBuf,

    #[arg(long, default_value_t = 2.0)]
    resolution: f64,

    #[arg(long, default_value_t = 50)]
    max_iterations: usize,

    #[arg(long, default_value_t = 0.5)]
    leaf_size: f64,

    #[arg(long)]
    initial_guess: Option<String>,
    
}

fn main() -> Result<()>{
    env_logger::init();
    let args = Args::parse();

    println!("Loading target point cloud from {:?}", args.target);
    println!("Loading source point cloud from {:?}", args.source);

    // rerun init (visualization)
    let rec = rerun::RecordingStreamBuilder::new("rsndt example - load and downsample point clouds")
        .spawn()
        .context("Failed to start rerun recording stream")?;

    println!("Loading point clouds...");
    let target_raw = load_pcd(&args.target)?;
    let source_raw = load_pcd(&args.source)?;

    println!("Target points: {}", target_raw.len());
    println!("Source points: {}", source_raw.len());

    // downsample
    println!("Downsampling (leaf_size = {})...", args.leaf_size);
    let mut voxel_filter = voxel_grid::VoxelGridFilter::new(args.leaf_size);
    let target_cloud = voxel_filter.filter(&target_raw);
    let source_cloud = voxel_filter.filter(&source_raw);

    println!("Downsampled Target points: {}", target_cloud.len());
    println!("Downsampled Source points: {}", source_cloud.len());

    log_point_cloud(&rec, "world/target", &target_cloud, [200, 200, 200])?;
    log_point_cloud(&rec, "world/source", &source_cloud, [255, 100, 100])?;

    // ndt
    println!("Performing NDT alignment (voxel resolution = {})...", args.resolution);
    let mut voxel_grid = voxel_grid::VoxelGrid::new(args.resolution);
    voxel_grid.set_input_cloud(&target_cloud);

    let mut matcher = NDTMatcher::new();
    matcher.set_max_iterations(args.max_iterations);
    let mut initial_guess = na::Isometry3::identity();
    if let Some(guess_str) = args.initial_guess {
        let vals: Vec<f64> = guess_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
            .context("Failed to parse initial guess")?;
        if vals.len() != 6 {
            return Err(anyhow::anyhow!("Initial guess must have 6 values: tx, ty, tz, roll, pitch, yaw"));
        }
        let translation = na::Translation3::new(vals[0], vals[1], vals[2]);
        let rotation = na::UnitQuaternion::from_euler_angles(vals[3], vals[4], vals[5]);
        initial_guess = na::Isometry3::from_parts(translation, rotation);
    }
    
    println!("Initial guess:\n{}", initial_guess);

    let _result = matcher.align(
        &source_cloud,
        &voxel_grid,
        initial_guess,
        Some(|iter: usize, pose: &na::Isometry3<f64>, score: f64| {
            println!("Iter {}: Score = {:.4}", iter, score);

            let translation = pose.translation.vector;
            let rotation = pose.rotation;

            rec.log(
                "world/source",
                &rerun::Transform3D::from_translation_rotation(
                    rerun::Vec3D::new(translation.x as f32, translation.y as f32, translation.z as f32),
                    rerun::Quaternion::from_xyzw([
                        rotation.i as f32,
                        rotation.j as f32,
                        rotation.k as f32,
                        rotation.w as f32,
                    ]),
                ),
            ).ok();
        })
    );

    println!("NDT alignment completed.");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
fn load_pcd(path: &PathBuf) -> Result<PointCloud> {
    let reader = Reader::<DynRecord, _>::open(path).context("Failed to open PCD file")?;
    let records: Result<Vec<DynRecord>, _> = reader.collect();

    let points: Result<Vec<Point>> = records?
        .into_iter()
        .enumerate()
        .map(|(idx, rec)| {
            let (x, y, z) = rec
                .to_xyz::<f32>()
                .map(|[x, y, z]| (x as f64, y as f64, z as f64))
                .or_else(|| rec.to_xyz::<f64>().map(|[x, y, z]| (x, y, z)))
                .ok_or_else(|| anyhow::anyhow!("PCD record #{idx} does not contain xyz fields"))?;
            Ok(Point::new(x, y, z))
        })
        .collect();
    
    Ok(PointCloud::new(points?))
}


fn log_point_cloud(rec: &rerun::RecordingStream, entity_path: &str, cloud: &PointCloud, color: [u8; 3]) -> Result<()> {
    let pos: Vec<[f32; 3]> = cloud.points.iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let colors: Vec<[u8; 4]> = vec![[color[0], color[1], color[2], 255]; cloud.len()];
    rec.log(
        entity_path,
        &rerun::Points3D::new(&pos)
            .with_colors(colors)
            .with_radii([0.1]),
    ).context("Failed to log points")?;
    Ok(())
}
