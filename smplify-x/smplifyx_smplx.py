import pickle
import torch
import smplx
import trimesh
import os

# === 경로 설정 ===
pkl_path = 'output_folder/results/COCO_val2014_000000000192/000.pkl'
model_folder = './models'  # SMPLX .npz 파일들이 있는 폴더
output_obj_path = 'output_folder/meshes/COCO_val2014_000000000192.obj'

# === 모델 로드 ===
model = smplx.create(
    model_path=model_folder,
    model_type='smplx',
    gender='neutral',
    ext='npz',
    use_pca=False,
    num_expression_coeffs=10,
    batch_size=1
).to('cpu')

# === .pkl 파일에서 파라미터 불러오기 ===
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)


from human_body_prior.tools.model_loader import load_vposer

# === VPoser 로드 ===
vposer, _ = load_vposer('./models/vposer/V02_05')

# === VPoser latent → axis-angle 복원 ===
body_pose_latent = torch.tensor(data['body_pose']).float().unsqueeze(0)
decoded = vposer.decode(body_pose_latent)
body_pose = decoded['pose_body']  # shape: (1, 21, 3)

left_hand_pose = data.get('left_hand_pose', np.zeros(45))
if left_hand_pose.shape[0] != 45:
    left_hand_pose = np.zeros(45)

right_hand_pose = data.get('right_hand_pose', np.zeros(45))
if right_hand_pose.shape[0] != 45:
    right_hand_pose = np.zeros(45)


# === SMPLX 모델에 파라미터 적용 ===
output = model(
    betas=torch.tensor(data['betas']).float().unsqueeze(0),
    global_orient=torch.tensor(data['global_orient']).float().unsqueeze(0),
    # body_pose=torch.tensor(data['body_pose']).float().unsqueeze(0),
    body_pose=body_pose,
    left_hand_pose=torch.tensor(data['left_hand_pose']).float().unsqueeze(0),
    right_hand_pose=torch.tensor(data['right_hand_pose']).float().unsqueeze(0),
    jaw_pose=torch.tensor(data['jaw_pose']).float().unsqueeze(0),
    leye_pose=torch.tensor(data['leye_pose']).float().unsqueeze(0),
    reye_pose=torch.tensor(data['reye_pose']).float().unsqueeze(0),
    expression=torch.tensor(data['expression']).float().unsqueeze(0),
    return_verts=True
)

# === 결과 저장 ===
vertices = output.vertices[0].detach().cpu().numpy()
faces = model.faces

mesh = trimesh.Trimesh(vertices, faces, process=False)
os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
mesh.export(output_obj_path)

print(f"✅ Saved mesh to {output_obj_path}")
