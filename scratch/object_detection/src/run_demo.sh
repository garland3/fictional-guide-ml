cd /workspaces/torch_tutorials_to_shared/scratch/object_detection
echo `pwd`
python video_read.py
cd /app/detectron2/demo/
echo `pwd`
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input /workspaces/torch_tutorials_to_shared/scratch/object_detection/imgs/image_name.jpg \
  --output /workspaces/torch_tutorials_to_shared/scratch/object_detection/outputs \
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl  \
        MODEL.DEVICE cpu