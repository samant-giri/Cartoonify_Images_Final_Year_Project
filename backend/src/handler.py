try:
    import unzip_requirements
except ImportError:
    pass

import json
from io import BytesIO
import time
import os
import base64

# import boto3
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, origins=['http://localhost:3000'])

def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str


def load_models():

    styles = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    models = {}

    for style in styles:
        model = Transformer()
        # response = s3.get_object(Bucket=bucket, Key=f"models/{style}_net_G_float.pth")
        state = torch.load(os.path.join("D:/Web Development Course/Cartoonify 2/cartoonify/backend/src/pretrained_models/", style + '_net_G_float.pth'))
        model.load_state_dict(state)
        model.eval()
        models[style] = model

    return models


gpu = -1

# s3 = boto3.client("s3")
# bucket = "cartoongan"

mapping_id_to_style = {0: "Hosoda", 1: "Hayao", 2: "Shinkai", 3: "Paprika"}

models = load_models()
print(f"models loaded ...")

@app.route("/cartoonify", methods=["POST"])
@cross_origin(origins=['http://localhost:3000'])
def cartoonify():
    """
    lambda handler to execute the image transformation
    """
    # warming up the lambda
    # if event.get("source") in ["aws.events", "serverless-plugin-warmup"]:
    #     print("Lambda is warm!")
    #     return {}

    data = request.json
    # print("data keys :", data.keys())
    image = data["image"]
    image = image[image.find(",") + 1 :]
    dec = base64.b64decode(image + "===")
    image = Image.open(BytesIO(dec))
    image = image.convert("RGB")

    # load the model with the selected style

    model_id = int(data["model_id"])
    load_size = int(data["load_size"])
    style = mapping_id_to_style[model_id]
    model = models[style]

    # resize the image

    h = image.size[0]
    w = image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    image = image.resize((h, w), Image.BICUBIC)
    image = np.asarray(image)

    # RGB -> BGR
    image = image[:, :, [2, 1, 0]]
    image = transforms.ToTensor()(image).unsqueeze(0)

    # preprocess, (-1, 1)
    image = -1 + 2 * image
    if gpu > -1:
        image = Variable(image, volatile=True).cuda()
    else:
        image = image.float()  # Variable(input_image).float()

    with torch.no_grad():
        output_image = model(image)
        output_image = output_image[0]

    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    output_image = output_image.numpy()

    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)

    output_image = Image.fromarray(output_image)

    #
    result = {"output": img_to_base64_str(output_image)}

    return {
        "statusCode": 200,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },


    }


if __name__ == "__main__":
    app.run(debug=True)