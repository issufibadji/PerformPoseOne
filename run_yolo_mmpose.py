# run_yolo_mmpose.py

import mmcv
from ultralytics import YOLO
# imports corrigidos:
from mmpose.apis import init_model, inference_topdown
from mmpose.apis.visualization import vis_pose_result


# 1) Configurações
IMG_PATH = 'tests/data/coco/000000000785.jpg'

# YOLOv8 tiny detection
YOLO_MODEL = 'yolov8n.pt'

# MMPose config + checkpoint (COCO 256×192)
POSE_CONFIG = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
POSE_CHECKPOINT = 'checkpoints/hrnet_w48_coco_256x192.pth'

DEVICE = 'cpu'
OUT_FILE = 'vis_results/coco_785_pipeline.jpg'

# 2) Detectar pessoas com YOLOv8
yolo = YOLO(YOLO_MODEL)
results = yolo.predict(source=IMG_PATH, device=DEVICE, save=False)[0]
bboxes = results.boxes.xyxy.cpu().numpy()
scores = results.boxes.conf.cpu().numpy()

# 3) Inicializar o modelo de pose
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)

# 4) Inferência top-down de pose
person_results = [{'bbox': b, 'bbox_score': s}
                  for b, s in zip(bboxes, scores)]
pose_results = inference_topdown(
    pose_model,
    IMG_PATH,
    person_results,
    format='xyxy',
    dataset='TopDownCocoDataset'
)

# 5) Visualizar e salvar só o esqueleto
img = mmcv.imread(IMG_PATH)
vis = vis_pose_result(
    pose_model,
    img,
    pose_results,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    show=False,
    draw_heatmap=False,
    alpha=1.0
)
mmcv.imwrite(vis, OUT_FILE)

print(f"✅ Pipeline concluído! Resultado salvo em {OUT_FILE}")
