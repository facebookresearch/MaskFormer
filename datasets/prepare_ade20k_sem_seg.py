#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import numpy as np
import tqdm
from PIL import Image

# Define converting function that will simply shift the label values
def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)

# Perform the conversion
if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS"), "ADEChallengeData2016")
    for name in ["training", "validation"]:
        annotation_dir = os.path.join(dataset_dir, "annotations", name)
        output_dir = os.path.join(dataset_dir, "annotations_detectron2", name)
        os.makedirs(output_dir, exist_ok=True)
        for file in tqdm.tqdm([x for x in os.listdir(annotation_dir) if x.endswith(".png")]):
            convert(os.path.join(annotation_dir, file), os.path.join(output_dir, file))
