import cv2
import numpy as np
import torch
from LoadModel import ImageModelN8
from components import processFrame, findBox, beastBox

PATH = "Models/ModelsN8.pth"
device = torch.device('cpu')
model = ImageModelN8()
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

labSize = 230
imgSize = 230


def trackPlayer(frame, box):
    (x, y, w, h) = [int(a) for a in box]
    frameBound = [y - 100, y + h + 100, x - 100, x + w + 100]

    cropped_image = frame[frameBound[0]:frameBound[1], frameBound[2]:frameBound[3]]

    # Model output
    frameInput = processFrame(cropped_image, imgSize)

    image = frameInput.to(device)
    output = model(image)

    output = output.view(imgSize, imgSize)
    threshold = 0.8
    outputTH = (output > threshold).float()

    outputTH = outputTH.cpu()  # Convert the image to cpu , to a numpy array
    outputTH = outputTH.detach().numpy()

    cropped_imageSize = cropped_image.shape
    resized_mask = cv2.resize(outputTH, (cropped_imageSize[1], cropped_imageSize[0]))
    canvas = np.zeros((frame.shape[0], frame.shape[1]))
    canvas[frameBound[0]:frameBound[1], frameBound[2]:frameBound[3]] = resized_mask * 255

    return canvas
