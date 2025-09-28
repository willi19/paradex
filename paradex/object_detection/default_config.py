from pathlib import Path

nas_path = Path.home()/'shared_data'/'202510_demo'

default_template = {
    "pringles": nas_path/"template/pringles/0",
    "brown_ramen_von" : nas_path/"template/brown_ramen_von/0",
    "red_ramen_von" : nas_path/"template/red_ramen_von/0",
    "yellow_ramen_von" : nas_path/"template/yellow_ramen_von/0"
}

name2prompt = {
    "pringles": "pringles",
    "brown_ramen_von" : "brownramen",
    "red_ramen_von" : "redramen",
    "yellow_ramen_von" : "yellowramen",
}

prompt2name = {value:key for key, value in name2prompt.items()}
