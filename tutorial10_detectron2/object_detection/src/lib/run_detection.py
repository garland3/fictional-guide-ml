# Code to actually run the detection
import requests
# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2


temp_img_file = "../imgs/test.jpg"
temp__output_img_file = "../outputs/test.jpg"

def get_image_from_host():
    r = requests.get("http://host.docker.internal:5000")
    img = r.content # it is downloading
    with open(temp_img_file,"wb") as f:
        f.write(img)
    return img

def setup_config():
    """Get configurations from a yaml file"""
    cfg = get_cfg()
    # f = "/workspaces/torch_tutorials_to_shared/tutorial10_detectron2/object_detection/config/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    f = "/workspaces/torch_tutorials_to_shared/tutorial10_detectron2/object_detection/config/COCO-InstanceSegmentation/my_masked_config.yaml"
    # cfg.merge_from_file(model_zoo.get_config_file(f))
    cfg.merge_from_file(f)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    return cfg

def make_predictor(cfg):
    return DefaultPredictor(cfg)

def predict_img(predictor, img):
    output  = predictor(img)
    return output

def run_visualizer(outputs, im, cfg):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    return out.get_image()  

if __name__ == "__main__":
    get_image_from_host()
    cfg = setup_config()
    pred = make_predictor(cfg)
    img = cv2.imread(temp_img_file)
    out_predictions = predict_img(pred, img)
    img_viz = run_visualizer(out_predictions,img, cfg)
    cv2.imwrite(temp__output_img_file, img_viz[:,:,::-1] )
