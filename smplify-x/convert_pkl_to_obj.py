# import pickle
# import os
# import numpy as np

# output_dir = 'output_folder/meshes/COCO_val2014_000000000192'
# pkl_path = 'output_folder/results/COCO_val2014_000000000192/000.pkl'

# os.makedirs(output_dir, exist_ok=True)

# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)

# vertices = data['vertices']
# faces = data['faces']

# obj_path = os.path.join(output_dir, '000.obj')
# with open(obj_path, 'w') as fp:
#     for v in vertices:
#         fp.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
#     for f in faces + 1:  # .obj 파일은 1-based indexing
#         fp.write('f {} {} {}\n'.format(f[0], f[1], f[2]))

# print(f'✅ Saved .obj to {obj_path}')


import pickle

pkl_path = 'output_folder/results/COCO_val2014_000000000192/000.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("✅ 타입:", type(data))
print("🔑 포함된 키 목록:")
for k in data.keys():
    print("  -", k)
