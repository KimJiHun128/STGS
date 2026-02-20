import numpy as np

masks_npy = "/home/jihun/PycharmProjects/SurgTPGS/data/cholecseg_sub/video01_00080/image_sam3_seg/00000/masks_stack.npy"
arr = np.load(masks_npy)  # expected (N,H,W)

print("shape:", arr.shape, "dtype:", arr.dtype)
print("min/max:", arr.min(), arr.max())

u = np.unique(arr)
print("unique 개수:", len(u))
print("앞 20개 unique:", u[:20])

# 판정
if set(u.tolist()).issubset({0, 1}):
    print("=> 0/1 바이너리 마스크")
elif set(u.tolist()).issubset({0, 255}):
    print("=> 0/255 바이너리 마스크")
else:
    print("=> 그레이스케일(또는 확률형) 마스크")
