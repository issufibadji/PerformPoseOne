# Resumo do Pipeline CPU-Only: Detecção → Pose Humana

## 1. Criar e ativar ambiente Conda

```bash
conda create -n pose_cpu_env python=3.10 -y
conda activate pose_cpu_env
```

## 2. Instalar PyTorch (CPU-only)

```bash
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

## 3. Preparar e instalar **mmcv-full** com extensões C++

```bash
# Instalar dependências de build e atualizar libstdc++
sudo apt update
sudo apt install -y build-essential cmake libpython3-dev ninja-build
conda install -y -c conda-forge libgcc-ng libstdcxx-ng

# Clonar, fazer checkout da tag v2.1.0 e instalar em modo editable
git clone https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv
git fetch --all --tags
git checkout v2.1.0
pip install -e .
```

## 4. Instalar MMEngine, MMPose e MMDetection

```bash
pip install mmengine mmpose mmdet
```

## 5. Instalar YOLOv8

```bash
pip install ultralytics
```

## 6. Baixar checkpoint HRNet-48 COCO 256×192

```bash
mkdir -p ~/mmpose/checkpoints
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
     -O ~/mmpose/checkpoints/hrnet_w48_coco_256x192.pth
```

## 7. Teste direto de estimativa de pose (Top-Down)

```bash
cd ~/mmpose
mkdir -p vis_results

python demo/image_demo.py \
  tests/data/300w/indoor_020.png \
  configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  checkpoints/hrnet_w48_coco_256x192.pth \
  --device cpu \
  --out-file vis_results/indoor_020_pose.jpg
```

## 8. Teste de detecção de pessoa (YOLOv8)

```bash
yolo detect predict \
  model=yolov8n.pt \
  source=tests/data/h36m/BF_IUV_gt/S1_Directions_1.54138969_000001_467_466.png \
  device=cpu \
  project=vis_results \
  name=yolov8_fullbody
```

## 9. Pipeline “Detecção → Pose” (Top-Down com RTMDet + HRNet)

```bash
export IMG=tests/data/h36m/BF_IUV_gt/S1_Directions_1.54138969_000001_467_466.png

python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  checkpoints/hrnet_w48_coco_256x192.pth \
  --device cpu \
  --input "$IMG" \
  --output-root vis_results/fullbody_yolo_mmpose \
  --draw-heatmap \
  --save-predictions
```

* **Imagem anotada:**
  `vis_results/fullbody_yolo_mmpose/S1_Directions_1.54138969_000001_467_466.png`
* **Keypoints JSON:**
  `vis_results/fullbody_yolo_mmpose/results_S1_Directions_1.54138969_000001_467_466.json`

---

Pronto! Esse arquivo Markdown contém o passo-a-passo completo para detectarmos a pessoa e estimarmos o esqueleto humano completo em CPU.

