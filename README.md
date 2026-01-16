
# rsndt

RustでNDT (Normal Distributions Transform) による点群位置合わせを試すためのsample実装

## Requirements

- Rust (cargo)
- rerun viewer（`rerun` クレートが自動で起動を試みます）

## Build

```bash
cargo build
```

## Run

```bash
cargo run -- \
	--target path/to/target.pcd \
	--source path/to/source.pcd \
	--resolution 2.0 \
	--leaf_size 0.5 \
	--max_iterations 50 \
	--neighbor_search N7 \
	--timing_csv out/timing.csv
```

- `--target`: 基準点群 (PCD)
- `--source`: 合わせ込みたい点群 (PCD)
- `--resolution`: NDTのボクセル解像度
- `--leaf_size`: ダウンサンプリング用のボクセルサイズ
- `--max_iterations`: NDTの最大反復回数
- `--neighbor_search`: 近傍探索方法（`Single` / `N7` / `N27`、未指定は `N7`）
- `--initial_guess`: 初期姿勢（`tx,ty,tz,roll,pitch,yaw`）
	- `roll/pitch/yaw` はラジアンを想定
- `--timing_csv`: 実行時間ログをCSVで保存（指定時のみ出力）

### Timing CSV

`--timing_csv` を指定すると、フェーズ時間とNDT各イテレーションの時間を1ファイルに出力される

- header: `kind,name,iter,elapsed_ms,delta_ms,score`
- `kind=phase`: フェーズ（例: `load_target_pcd`, `downsample`, `build_voxel_grid`, `ndt_alignment`）
- `kind=iter`: NDTイテレーション
	- `elapsed_ms`: NDT開始からの経過
	- `delta_ms`: 前回コールバックからの差分
	- `score`: NDTスコア

## Input PCD format

PCDのスキーマ差分（`XYZ` / `XYZI` など）に耐えるため、読み込みは `pcd-rs` の `DynRecord` を使って `x/y/z` を抽出

注意:
- `FIELDS` の先頭3つが `x y z` であることを想定
- それ以外（例: `rgb x y z` の順など）のPCDは追加対応が必要

## Notice
- 現在追加実装中…　実用には程遠い

