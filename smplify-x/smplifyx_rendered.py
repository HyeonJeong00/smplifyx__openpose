import cv2
import pickle
import torch
import numpy as np
from smplx import SMPLX
import smplx

# === 경로 설정 ===
img_path = 'data_folder/images/COCO_val2014_000000000192.jpg'
pkl_path = 'output_folder/results/COCO_val2014_000000000192/000.pkl'
model_folder = './models'

# === 원본 이미지 불러오기 ===
img = cv2.imread(img_path)

# === .pkl 파일 로딩 ===
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# === SMPLX 모델 생성 ===
model = smplx.create(
    model_folder,
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False,  # ✨ 손 포즈를 axis-angle 형식으로 받을 거라면 꼭 필요!
    num_expression_coeffs=10,  # <- 표정 파라미터도 있으면 같이 설정
    batch_size=1
).to('cpu')

left_hand_pose = data.get('left_hand_pose', np.zeros(45))
if left_hand_pose.shape[0] != 45:
    print(f"[경고] left_hand_pose 크기 이상함: {left_hand_pose.shape}, 0으로 패딩합니다.")
    left_hand_pose = np.zeros(45)

right_hand_pose = data.get('right_hand_pose', np.zeros(45))
if right_hand_pose.shape[0] != 45:
    print(f"[경고] right_hand_pose 크기 이상함: {right_hand_pose.shape}, 0으로 패딩합니다.")
    right_hand_pose = np.zeros(45)
betas = torch.tensor(data['betas']).float()
if betas.ndim == 1:  # (10,)
    betas = betas.unsqueeze(0)  # → (1, 10)
expression = torch.tensor(data['expression']).float()
if expression.ndim == 1:
    expression = expression.unsqueeze(0)  # (10,) -> (1, 10)
elif expression.ndim == 3:
    expression = expression.squeeze()  # (1, 10, 1) → (1, 10)

if expression.ndim == 2 and expression.shape[0] == 1 and expression.shape[1] == 10:
    pass  # OK!
else:
    print("[오류] expression의 shape가 예상과 다름:", expression.shape)

output = model(
    betas=betas,
    global_orient=torch.tensor(data['global_orient']).float().unsqueeze(0),
    body_pose=torch.tensor(data['body_pose']).float().unsqueeze(0),
    left_hand_pose=torch.tensor(left_hand_pose).float().unsqueeze(0),
    right_hand_pose=torch.tensor(right_hand_pose).float().unsqueeze(0),
    jaw_pose=torch.tensor(data['jaw_pose']).float().unsqueeze(0),
    leye_pose=torch.tensor(data['leye_pose']).float().unsqueeze(0),
    reye_pose=torch.tensor(data['reye_pose']).float().unsqueeze(0),
    expression=expression,
    return_verts=True
)

joints_3d = output.joints.detach().cpu().numpy().squeeze()
camera_translation = data['camera_translation']
focal_length = 5000  
joints_3d_cam = joints_3d + np.array(camera_translation).reshape(1, 3)

# === projection: perspective
joints_2d = focal_length * (joints_3d_cam[:, :2] / joints_3d_cam[:, 2:])  # [N, 2]
joints_2d += np.array([[img.shape[1] / 2, img.shape[0] / 2]])  # 중심 이동

# === 디버그 시각화
H, W = img.shape[:2]

# SMPL-X의 127개 joint 중 앞쪽 55개 이름 리스트
joint_names = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index1',
    'right_index1', 'left_index2', 'right_index2', 'left_middle1', 'right_middle1',
    'left_middle2', 'right_middle2', 'left_pinky1', 'right_pinky1', 'left_pinky2',
    'right_pinky2', 'left_ring1', 'right_ring1', 'left_ring2', 'right_ring2',
    'left_thumb1', 'right_thumb1', 'left_thumb2', 'right_thumb2',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index3', 'right_index3',
    'left_middle3', 'right_middle3', 'left_pinky3', 'right_pinky3',
    'left_ring3', 'right_ring3', 'left_thumb3', 'right_thumb3'
]


for i, (x, y) in enumerate(joints_2d):
    if 0 <= x < W and 0 <= y < H:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
        label = joint_names[i] if i < len(joint_names) else str(i)
        cv2.putText(img, str(i), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)
    else:
        print(f"⚠️ joint {i} is out of image bounds: ({x:.1f}, {y:.1f})")


# === 결과 저장
save_path = 'output_folder/rendered_overlay.jpg'
cv2.imwrite(save_path, img)
print(f"✅ Saved projected joints to {save_path}")

print(model.J_regressor.shape)  # ex. (127, 10475)

# skeleton_connections = [
#     (0, 3),    # pelvis → spine1
#     (3, 6),    # spine1 → neck
#     (6, 9),    # neck → head

#     (3, 13), (13, 14), (14, 15),     # left arm
#     (3, 16), (16, 17), (17, 18),     # right arm

#     (0, 1), (1, 4), (4, 7),          # left leg (pelvis → hip → knee → ankle)
#     (0, 2), (2, 5), (5, 8),          # right leg
# ]
skeleton_connections = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # spine → neck → head

    (13, 16), (16, 18), (18, 20),  # left arm
    (14, 17), (17, 19), (19, 21),  # right arm

    (0, 1), (1, 4), (4, 7), (7, 10),  # left leg
    (0, 2), (2, 5), (5, 8), (8, 11)   # right leg
]



for x, y in joints_2d:
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

for i, j in skeleton_connections:
    if i < joints_2d.shape[0] and j < joints_2d.shape[0]:
        pt1 = tuple(np.round(joints_2d[i]).astype(int))
        pt2 = tuple(np.round(joints_2d[j]).astype(int))
        if all(0 <= pt1[k] < img.shape[1 - k] and 0 <= pt2[k] < img.shape[1 - k] for k in [0, 1]):
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

# 저장
cv2.imwrite('output_folder/rendered_overlay_skeleton.jpg', img)



# import pickle
# import torch
# import numpy as np
# import smplx
# import trimesh
# import os

# pkl_path = 'output_folder/results/COCO_val2014_000000000192/000.pkl'
# model_folder = './models'
# output_obj_path = 'output_folder/meshes/COCO_val2014_000000000192/000.obj'

# # === .pkl 파일 로드
# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)
# # === OBJ 저장
# vertices = output.vertices[0].detach().cpu().numpy()
# faces = model.faces

# os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
# mesh = trimesh.Trimesh(vertices, faces, process=False)
# mesh.export(output_obj_path)

# print(f"✅ 저장 완료: {output_obj_path}")