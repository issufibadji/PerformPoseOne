# Guia de Instalação do Ambiente — GPU vs CPU

Este documento fornece **duas** opções de configuração do ambiente para estimativa de pose humana e detecção: uma **com GPU** (setup streamlined com wheels pré-compilados) e uma **alternativa para CPU-only** (adequada onde não há GPU disponível).

---

## Parte A: Instalação Completa para GPU

### 1. Pré-requisitos

* GPU NVIDIA com suporte CUDA (compute capability ≥3.5)
* Drivers NVIDIA e CUDA Toolkit instalados (ex.: CUDA 11.7)
* Ubuntu 20.04 LTS ou 22.04 LTS
* Conda (Miniconda ou Anaconda)

### 2. Criar e ativar ambiente Conda

```bash
conda create -y -n pose_gpu_env python=3.10
conda activate pose_gpu_env
```

### 3. Instalar PyTorch com CUDA

```bash
conda install -y pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch
```

### 4. Instalar MMCV-full compatível com CUDA

```bash
pip install mmcv-full==2.2.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html
```

### 5. Instalar MMEngine, MMPose e MMDetection

```bash
pip install mmengine mmpose mmdet
```

### 6. (Opcional) RTMPose Demo Completo

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
# O projeto RTMPose está dentro de projects/rtmpose
cd projects/rtmpose
pip install -e .
cd ../../

# Execute demo de detecção + pose (RTMDet + RTMPose)
python demo/topdown_demo_with_mmdet.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  checkpoints/hrnet_w48_coco_256x192.pth \
  --device cuda:0 \
  --show
```

### 7. Instalar e testar YOLOv8

```bash
pip install ultralytics
# Detectar pessoas em GPU
yolo detect predict model=yolov8n.pt source=tests/data/300w/indoor_020.png device=0 save-txt=True save-conf=True
```

### 8. Pipeline YOLOv8 → MMPose (em Python)

```bash
# Salve script run_yolo_mmpose.py (exemplo no doc original)
python run_yolo_mmpose.py --device cuda:0
```

### 9. Exportar ambiente

```bash
conda env export > pose_gpu_env.yaml
```

---

## Parte B: Alternativa para CPU-only

### 1. Pré-requisitos

* Sem GPU ou sem drivers CUDA
* Ubuntu 20.04 LTS ou 22.04 LTS
* Conda (Miniconda ou Anaconda)

### 2. Criar e ativar ambiente Conda

```bash
conda create -y -n pose_cpu_env python=3.10
conda activate pose_cpu_env
```

### 3. Instalar PyTorch CPU-only

```bash
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
```

### 4. Instalar mmcv-lite (pure Python) ou compilar mmcv-full

**Opção A (mmcv-lite)**

```bash
pip install mmcv-lite==2.1.0
```

**Opção B (mmcv-full compile from source)**

```bash
# Dependências de build
sudo apt update && sudo apt install -y build-essential cmake libpython3-dev ninja-build
# Clone e compile
git clone https://github.com/open-mmlab/mmcv.git ~/mmcv
cd ~/mmcv && git checkout v2.1.0 && pip install -e .
```

### 5. Atualizar libstdc++ (para ops C++)

```bash
conda install -y -c conda-forge libgcc-ng libstdcxx-ng
```

### 6. Instalar MMEngine, MMPose e opcionalmente MMDetection

```bash
pip install mmengine mmpose
# Para demos detect+pose
pip install mmdet
```

### 7. (Opcional) RTMPose via rtmlib (ONNX)

```bash
pip install rtmlib
```

### 8. Instalar YOLOv8

```bash
pip install ultralytics
```

### 9. Verificar e testar para **estimativa de pose humana**

#### 9.1 Teste MMPose (Top-Down) com imagem de pessoa

```bash
python demo/image_demo.py \
  tests/data/300w/indoor_020.png \
  configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  checkpoints/hrnet_w48_coco_256x192.pth \
  --device cpu \
  --out-file vis_results/indoor_020_pose.jpg
```

#### 9.2 Teste RTMPose via rtmlib com pessoa

```bash
python - << EOF
import cv2
from rtmlib import Wholebody, draw_skeleton
img = cv2.imread('tests/data/300w/indoor_020.png')
model = Wholebody(backbone='rtmpose-m', device='cpu')
res = model.predict(img)
vis = draw_skeleton(img, res)
cv2.imwrite('vis_results/rtmpose_indoor_020.jpg', vis)
EOF
```

#### 9.3 Teste YOLOv8 para detecção de pessoa

```bash
yolo detect predict model=yolov8n.pt source=tests/data/300w/indoor_020.png device=cpu save-txt=True save-conf=True project=vis_results name=yolov8_indoor
```

### 10. Exportar ambiente

```bash
conda env export > pose_cpu_env.yaml
```

---

Com esses guias, o foco é **estimativa de pose humana**, utilizando exemplos de imagem com pessoas (`tests/data/300w/indoor_020.png`), sem referência a cavalos. Escolha GPU ou CPU conforme disponibilidade do hardware.

