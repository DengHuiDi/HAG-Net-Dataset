# Dataset A fixed split manifest

[`dataset_a_manifest.csv`](dataset_a_manifest.csv) identifies the images in the fixed Dataset A split used for the main experiments: 4,734 training images and 1,639 validation images (6,373 total). This release records the existing split; it does not create or change a split. The YOLO class order is class 0 = `fire` and class 1 = `smoke`.

Each CSV row represents one image. `image_id` is the image stem; `relative_image_path` and `relative_label_path` are text identifiers relative to a Dataset A root; and `split` is `train` or `val`. `image_sha256` is the SHA-256 digest of the image file, provided to check file identity against an independently obtained copy. The CSV does not contain image bytes or annotation contents.

`source_dataset` records the dataset-organisation category. `source_attribution_status` and `source_attribution_basis` state the evidence level and rule used for that category. All current source assignments are marked `probable` because they follow the authors' recorded naming and organisation rules; they are not represented as independently verified original-file matches.

No Dataset A images, YOLO label files, previews, or archives are distributed with this manifest. The manifest supports sample identification and split traceability, but cannot reconstruct the dataset by itself and does not grant permission to redistribute source material. Users must obtain any underlying data through methods permitted by the respective providers. This repository does not assert an unverified download URL, licence, or access procedure for Dataset A.
