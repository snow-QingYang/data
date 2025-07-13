import json
import os


if __name__ == "__main__":
    data_root = "/home/huanghz/space/code/diffusion12/preprocess/coyo_subset/coyo-700M-filtered-r"
    json_path = "/home/huanghz/space/code/diffusion12/preprocess/coyo_subset/coyo-700M-filtered-r-index.json"

    output_path = ""

    with open(json_path, "r") as f:
        data = json.load(f)
    for item in data:
        key, instance_id = item['key'], item['instance_id']
        image_path = os.path.join(data_root, f"{key}.jpg")
        glb_path = os.path.join(data_root)
        depth_path = os.path.join(data_root)
