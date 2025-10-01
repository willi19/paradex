from pathlib import Path

project_path = Path(__file__).absolute().parent.parent
nas_path = Path.home()/'shared_data'/'object_6d'/'data'

default_template = {
    "pringles": nas_path/"template/pringles/0",
    "brown_ramen_von_2" : nas_path/"template/brown_ramen_von_2/0",
    "red_ramen_von_2" : nas_path/"template/red_ramen_von_2/0",
    "yellow_ramen_von_2" : nas_path/"template/yellow_ramen_von_2/0"
}

name2prompt = {
    "pringles": "pringles",
    "brown_ramen_von_2" : "brownramen",
    "red_ramen_von_2" : "redramen",
    "yellow_ramen_von_2" : "yellowramen",
}

prompt2name = {value:key for key, value in name2prompt.items()}

yolo_pretrained_path = nas_path/'checkpoint'/'best_v4.pt'
ELOFTR_CKPT_PATH = project_path/'thirdparty'/'EfficientLoFTR'/'ckpts'/'eloftr_outdoor.ckpt'
