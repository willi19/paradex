from pathlib import Path

project_path = Path(__file__).absolute().parent.parent.parent
nas_path = Path.home() / "shared_data" / "object_6d" / "data"

default_template = {
    "pringles": nas_path / "template/pringles/0",
    # "brown_ramen_von_2": nas_path / "template/brown_ramen_von_2/0",
    # "red_ramen_von_2": nas_path / "template/red_ramen_von_2/0",
    # "yellow_ramen_von_2": nas_path / "template/yellow_ramen_von_2/0",
    "brown_ramen_von" : nas_path/"template/brown_ramen_von/0",
    "red_ramen_von" : nas_path/"template/red_ramen_von/0",
    "yellow_ramen_von" : nas_path/"template/yellow_ramen_von/0"
}

template2camids = {
    "brown_ramen_von" : [24122734,25305461,25305462,25305463,25305464,25305467,25322642,25322644,25322649,25322651,25305467,],
    "red_ramen_von" : [25305461,25305462,25305463,25305464,25322644,25322646,25322648,25322649,25322650,25322651],
    "yellow_ramen_von" : [24122734,25305460,25305461,25305462,25305464,25305465,25322639,25322644,25322646,25322649,25322651]
}


name2prompt = {
    "pringles": "pringles",
    "brown_ramen_von_2": "brownramen",
    "red_ramen_von_2": "redramen",
    "yellow_ramen_von_2": "yellowramen",
    "brown_ramen_von" : "brownramen",
    "red_ramen_von" : "redramen",
    "yellow_ramen_von" : "yellowramen",
}

prompt2name = {value: key for key, value in name2prompt.items()}

model_path = {
    'customized_sam':nas_path/'checkpoint'/'best_v4.pt',
    'efficient_sam':nas_path/'checkpoint'/'efficient_sam_s_gpu.jit',
    'eloftr':nas_path/'checkpoint'/'eloftr_outdoor.ckpt',
}

yolo_pretrained_path = model_path["customized_sam"]
ELOFTR_CKPT_PATH = model_path["eloftr"]
