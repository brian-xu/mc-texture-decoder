import json
import os

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

MC_VERSION = "1.20.2"
PCA_COMPONENTS = 72


block_collision_shapes_json = (
    f"data/minecraft-data/data/pc/{MC_VERSION}/blockCollisionShapes.json"
)

with open(block_collision_shapes_json) as f:
    block_collision_shapes = json.loads(f.read())["blocks"]

solid_blocks = []

for block in block_collision_shapes:
    if block_collision_shapes[block] == 1:
        solid_blocks.append(block)


block_models_json = f"data/minecraft-assets/data/{MC_VERSION}/blocks_models.json"

with open(block_models_json) as f:
    block_models = json.loads(f.read())

textures_root = f"data/minecraft-assets/data/{MC_VERSION}/blocks"

valid_blocks = {}

for block in solid_blocks:
    if block in block_models:
        texture_path = None
        for key in ["all", "side", "pattern", "texture"]:
            if key in block_models[block]["textures"]:
                texture_path = block_models[block]["textures"][key].split("/")[1]
                break
        if texture_path:
            block_texture_path = os.path.join(textures_root, texture_path + ".png")
            valid_blocks[block] = block_texture_path


labels = []
image_data = []

for block in valid_blocks:
    labels.append(block)
    image = Image.open(valid_blocks[block]).convert("RGB")
    im_array = np.asarray(image)[:16, :16, :]
    image_data.append(im_array)

image_data = np.array(image_data) / 255.0

pca = PCA(n_components=PCA_COMPONENTS)
data_reduced = pca.fit_transform(np.reshape(image_data, (image_data.shape[0], -1)))

embeddings = (data_reduced - data_reduced.min()) / (
    data_reduced.max() - data_reduced.min()
)

with open("processed/labels.json", "w") as f:
    json.dump(labels, f)
np.save("processed/images", image_data)
np.save("processed/embeddings", embeddings)
