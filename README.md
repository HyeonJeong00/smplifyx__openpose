SMPLX 에 직접 가서 다운받은 후, models 파일을 만들고 직접 넣어야함!
용량이 커서 잘 안올라갔음
https://smpl-x.is.tue.mpg.de/

smplifyx/models/smplx/SMPLX_FEMALE.npz
smplifyx/models/smplx/SMPLX_FEMALE.pkl
smplifyx/models/smplx/SMPLX_MALE.npz
smplifyx/models/smplx/SMPLX_MALE.pkl
smplifyx/models/smplx/SMPLX_NEUTRAL.npz
smplifyx/models/smplx/SMPLX_NEUTRAL.pkl
smplifyx/models/smplx/smplx_parts_segm.pkl

vposer도 위의 링크에 있는 version2를 다운 받은 후에 이런 식으로 잘 넣어야함.
smplifyx/models/vposer/V02_05/snapshot/ckpt_pth.py
smplifyx/models/vposer/V02_05/snapshot/epoch_20.pth


python smplifyx/main.py \
  --config cfg_files/fit_smplx.yaml \
  --data_folder ./data_folder \
  --output_folder ./output_folder \
  --visualize=False \
  --save_meshes=True \
  --model_folder ./models \
  --vposer_ckpt ./models/vposer/V02_05 \
  --part_segm_fn ./models/smplx/smplx_parts_segm.pkl

version은 smplifyx repo

conda create -n smplxxx python=3.8 -y
conda activate smplxxx

conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install numpy opencv-python tqdm pyrender smplx configargparse PyYAML
pip install git+https://github.com/nghorbani/human_body_prior.git#egg=human_body_prior

conda install -c conda-forge omegaconf
conda install -c conda-forge loguru
conda install -c conda-forge dotmap
