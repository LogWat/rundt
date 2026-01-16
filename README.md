
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
	--leaf_size 0.5
```

- `--target`: 基準点群 (PCD)
- `--source`: 合わせ込みたい点群 (PCD)
- `--resolution`: NDTのボクセル解像度
- `--leaf_size`: ダウンサンプリング用のボクセルサイズ

## Input PCD format

PCDのスキーマ差分（`XYZ` / `XYZI` など）に耐えるため、読み込みは `pcd-rs` の `DynRecord` を使って `x/y/z` を抽出

注意:
- `FIELDS` の先頭3つが `x y z` であることを想定
- それ以外（例: `rgb x y z` の順など）のPCDは追加対応が必要

## Notice
- 現在追加実装中…　実用には程遠い

