# run_yolo_mmpose.py

import mmcv
from ultralytics import YOLO
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result

# 1) Configurações
IMG_PATH = 'tests/data/300w/indoor_020.png'

# YOLOv8 tiny detection
YOLO_MODEL = 'yolov8n.pt'

# MMPose config + checkpoint (COCO 256×192)
POSE_CONFIG = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
POSE_CHECKPOINT = 'checkpoints/hrnet_w48_coco_256x192.pth'

DEVICE = 'cpu'
OUT_FILE = 'vis_results/indoor_020_pipeline.jpg'

# 2) Detectar pessoas com YOLOv8
yolo = YOLO(YOLO_MODEL)
results = yolo.predict(source=IMG_PATH, device=DEVICE, save=False)[0]
# extrai boxes no formato [x1, y1, x2, y2]
bboxes = results.boxes.xyxy.cpu().numpy()

# 3) Inicializar o modelo de pose
pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)

# 4) Inferência top-down de pose
# MMPose espera lista de dicts com chave 'bbox'
bboxes_mmpose = [{'bbox': box, 'bbox_score': score}
                 for box, score in zip(bboxes, results.boxes.conf.cpu().numpy())]

pose_results = inference_top_down_pose_model(
    pose_model,
    IMG_PATH,
    bboxes_mmpose,
    format='xyxy',
    dataset='TopDownCocoDataset'
)

# 5) Visualizar e salvar
img = mmcv.imread(IMG_PATH)
vis = vis_pose_result(
    pose_model, img, pose_results,
    radius=4, thickness=1,
    kpt_score_thr=0.3, show=False
)
mmcv.imwrite(vis, OUT_FILE)

print(f"✅ Pipeline concluído! Resultado salvo em {OUT_FILE}")

