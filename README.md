# PerformPoseOne
Perform Pose

## Adaptar para 12 pontos-chave
Este repositório inclui um exemplo de configuração do MMPose
para usar apenas **12 pontos-chave**. O novo arquivo de metainfo
fica em `configs/_base_/datasets/coco_12.py` e a configuração de
treinamento correspondente está em
`configs/body_2d_keypoint/topdown_heatmap/custom/td-hm_hrnet-w48_8xb32-210e_custom12-256x192.py`.

Para utilizar o pipeline com 12 pontos:

1. Converta suas anotações para o formato COCO contendo apenas as
   12 chaves definidas em `coco_12.py`.
2. Treine o modelo usando a configuração personalizada acima ou
   utilize os scripts de demonstração que filtram o modelo COCO
   pré-treinado para apenas 12 pontos.
3. Execute `run_yolo_mmpose.py` ou `run_yolo_mmpose1.py`. Eles
   carregam o modelo COCO de 17 pontos e convertem as predições
   para o conjunto de 12 pontos definido em `coco_12.py`.
