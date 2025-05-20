import h5py
import numpy as np

# H5 파일 경로
h5_path = 'data/auto_labeled_patient_dataset.h5'

# 파일 열기 및 라벨 읽기
with h5py.File(h5_path, 'r') as f:
    labels = f['label'][:].squeeze()  # (N, 1) -> (N,)으로 평탄화

# 라벨 분포 출력
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Label {int(label)}: {count} samples")
