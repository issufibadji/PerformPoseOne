# run_yolo_mmpose.py

import mmcv
from ultralytics import YOLO
# imports corrigidos:
from mmpose.apis import init_model, inference_topdown, vis_pose_result


# 1) Configurações
IMG_PATH = 'tests/data/coco/000000000785.jpg'

# YOLOv8 tiny detection
YOLO_MODEL = 'yolov8n.pt'

# MMPose config + checkpoint (COCO 256×192)
POSE_CONFIG = 'configs/body_2d_keypoint/topdown_heatmap/custom/td-hm_hrnet-w48_8xb32-210e_custom12-256x192.py'
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
# A função ``inference_topdown`` espera receber apenas as caixas de
# delimitação (bboxes) ou ``None``. A pontuação de detecção não é
# necessária, portanto passamos somente ``bboxes``.
pose_results = inference_topdown(
    pose_model,
    IMG_PATH,
    bboxes,
    bbox_format='xyxy'
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
