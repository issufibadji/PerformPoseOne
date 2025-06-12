# run_yolo_mmpose.py

import mmcv
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown, vis_pose_result
from mmpose.datasets.datasets.utils import parse_pose_metainfo


# 1) Configurações
IMG_PATH = 'tests/data/coco/000000000785.jpg'

# YOLOv8 tiny detection
YOLO_MODEL = 'yolov8n.pt'

# MMPose config + checkpoint (COCO 256×192)
POSE_CONFIG = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
POSE_CHECKPOINT = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

DEVICE = 'cpu'
OUT_FILE = 'vis_results/coco_785_pipeline.jpg'

# Índices dos 12 pontos que queremos (ombros, cotovelos, punhos,
# quadris, joelhos e tornozelos) no modelo COCO de 17 pontos
COCO_TO_CUSTOM12 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

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

# Filtra as chaves do modelo COCO para apenas 12 pontos
for res in pose_results:
    res.pred_instances.keypoints = res.pred_instances.keypoints[:, COCO_TO_CUSTOM12, :]
    res.pred_instances.keypoint_scores = res.pred_instances.keypoint_scores[:, COCO_TO_CUSTOM12]

# Usa metainfo de 12 pontos apenas para visualização
pose_model.dataset_meta = parse_pose_metainfo(
    dict(from_file='configs/_base_/datasets/coco_12.py'))

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
