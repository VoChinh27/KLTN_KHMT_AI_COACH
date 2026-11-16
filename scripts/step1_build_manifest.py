import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1_data.dataset_builder import DatasetBuilder, DatasetBuilderConfig

builder = DatasetBuilder(DatasetBuilderConfig(
    raw_video_dir="data/raw_videos",
    label_store_path="data/labels.jsonl",
    manifest_out_path="data/manifest.csv",
    test_ratio=0.2,
))
manifest = builder.build()
builder.save_manifest(manifest)
print(builder.summary(manifest))