import numpy as np
from PIL import Image
from exit import ExitEx
from helper_classes import Pair
import os

def parse(filename):
    with Image.open(filename) as img:
        pixel_data = list(img.getdata())
        if img.mode == "RGB":
            pixel_data = [pixel[:3] for pixel in pixel_data]
        return [pixel_data[i:i + img.width] for i in range(0, len(pixel_data), img.width)]

def parse_bmp(filename):
    pixel_data = parse(filename)

    walls = []
    exits = []

    tmp_exit = ExitEx(0)
    for y, row in enumerate(pixel_data):
        for x, pixel in enumerate(row):
            if pixel == (0, 0, 0):
                walls.append(Pair(x, y))
            elif pixel == (0, 0, 255):
                if (tmp_exit.is_empty()):
                    tmp_exit.id = len(exits)
                    tmp_exit.add_point(Pair(x, y))
                else:
                    if (tmp_exit.points[-1].x == x) or (tmp_exit.points[-1].y == y):
                        tmp_exit.add_point(Pair(x, y))
                    else:
                        exits.append(tmp_exit)
                        tmp_exit = ExitEx(len(exits))
                        tmp_exit.add_point(Pair(x, y))
    
    if not tmp_exit.is_empty():
        exits.append(tmp_exit)
                    
    return exits, walls, Pair(len(pixel_data[0]), len(pixel_data))