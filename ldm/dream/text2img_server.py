import argparse
from datetime import datetime
import json
import shutil
from urllib.parse import urlparse
from ldm.dream.model_leader import get_model
from ldm.db_logger import addQuery, getQueries
import torch
from PIL import Image
import numpy as np
import io
import PIL
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 3
default_model = "stable-diffusion-1.4"

def build_opt(prompt):
    opt = argparse.Namespace()
    setattr(opt, 'prompt', prompt)
    setattr(opt, 'init_img', None)
    setattr(opt, 'strength',  0.75)
    setattr(opt, 'iterations', 1)
    setattr(opt, 'steps', 50)
    setattr(opt, 'width', 512)
    setattr(opt, 'height', 512)
    setattr(opt, 'seamless', False)
    setattr(opt, 'fit', True)
    setattr(opt, 'mask', False)
    setattr(opt, 'invert_mask', False)
    setattr(opt, 'cfg_scale', 7.5)
    setattr(opt, 'sampler_name', 'k_lms')
    setattr(opt, 'gfpgan_strength', 0)
    setattr(opt, 'upscale', None)
    setattr(opt, 'progress_images', False)
    setattr(opt, 'seed', None)
    setattr(opt, 'variation_amount', 0)
    setattr(opt, 'with_variations', None)
    return opt

def build_opt2(prompt, strength, steps):
    opt = argparse.Namespace()
    setattr(opt, 'prompt', prompt)
    setattr(opt, 'init_img', None)
    setattr(opt, 'strength',  strength)
    setattr(opt, 'iterations', 1)
    setattr(opt, 'steps', steps)
    setattr(opt, 'width', 512)
    setattr(opt, 'height', 512)
    setattr(opt, 'seamless', False)
    setattr(opt, 'fit', True)
    setattr(opt, 'mask', False)
    setattr(opt, 'invert_mask', False)
    setattr(opt, 'cfg_scale', 7.5)
    setattr(opt, 'sampler_name', 'k_lms')
    setattr(opt, 'gfpgan_strength', 0)
    setattr(opt, 'upscale', None)
    setattr(opt, 'progress_images', False)
    setattr(opt, 'seed', None)
    setattr(opt, 'variation_amount', 0)
    setattr(opt, 'with_variations', None)
    return opt

def hex_color_string_to_tuple(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_img(w, h, color):
    image = Image.new('RGB', (w, h), color)
    return image

def convert_img(image):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def renderImage(self, data, _model):
    _model.load_textual_inversion_embeddings()
    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header('Content-type', 'image/png')
    self.end_headers()
    
    def image_done(image, seed, upscaled=False):
        addQuery('txt2img', image, data, None, opt.strength, opt.iterations, opt.steps, opt.width, opt.height, opt.seamless, opt.fit, opt.mask, opt.invert_mask, opt.cfg_scale, opt.sampler_name, opt.gfpgan_strength, opt.upscale, opt.progress_images, opt.seed, opt.variation_amount, opt.with_variations)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        self.wfile.write(img_byte_arr.getvalue())
    
    def image_progress(sample, step):
        pass

    opt = build_opt(data)
    _model.prompt2image(**vars(opt), step_callback=image_progress, image_callback=image_done)


def renderImageMass(self, data, _model, count):
    _model.load_textual_inversion_embeddings()
    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header('Content-type', 'application/zip')
    self.end_headers()

    folderName = datetime.now().strftime("%Y%m%d%H%M%S") + "_temp"
    os.mkdir(folderName)

    def image_done(image, seed, currentCount, upscaled=False):
        addQuery('txt2img', image, data, None, opt.strength, opt.iterations, opt.steps, opt.width, opt.height, opt.seamless, opt.fit, opt.mask, opt.invert_mask, opt.cfg_scale, opt.sampler_name, opt.gfpgan_strength, opt.upscale, opt.progress_images, opt.seed, opt.variation_amount, opt.with_variations)
        image.save(folderName + '/' + str(currentCount) + '.png', format='PNG')
        if (currentCount >= count):
            zipFileName = datetime.now().strftime("%Y%m%d%H%M%S") + 'images'
            shutil.make_archive(zipFileName, 'zip', folderName)
            for root, dirs, files in os.walk(folderName, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(folderName)
            _data = []
            with open(zipFileName + '.zip', 'rb') as f:
                _data = io.BytesIO(f.read())
            
            os.remove(zipFileName + '.zip')
            self.wfile.write(_data.read())
            return

        _model.prompt2imageWithCount(**vars(opt), step_callback=image_progress, image_callback=image_done, currentCount=currentCount)

    
    def image_progress(sample, step):
        pass

    opt = build_opt(data)
    _model.prompt2imageWithCount(**vars(opt), step_callback=image_progress, image_callback=image_done, currentCount=0)


def renderModImage(self, init_image, data, strength, steps, _model):  
    _model.load_textual_inversion_embeddings()
    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header('Content-type', 'image/png')
    self.end_headers()

    def image_done(image, seed, upscaled=False):
        addQuery('img2img', image, data, init_image, opt.strength, opt.iterations, opt.steps, opt.width, opt.height, opt.seamless, opt.fit, opt.mask, opt.invert_mask, opt.cfg_scale, opt.sampler_name, opt.gfpgan_strength, opt.upscale, opt.progress_images, opt.seed, opt.variation_amount, opt.with_variations)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        self.wfile.write(img_byte_arr.getvalue())
        print('generated image, end')
    
    def image_progress(sample, step):
        pass

    try:
        init_image.save("./img2img-tmp.png")
        opt = build_opt2(data, strength, steps)
        opt.init_img = "./img2img-tmp.png"
        _model.prompt2image(**vars(opt), step_callback=image_progress, image_callback=image_done)
    finally:
        os.remove("./img2img-tmp.png")

def renderModImageMass(self, init_image, data, strength, steps, _model, count):
    _model.load_textual_inversion_embeddings()
    self.send_response(200)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header('Content-type', 'application/zip')
    self.end_headers()

    folderName = datetime.now().strftime("%Y%m%d%H%M%S") + "_temp"
    os.mkdir(folderName)

    def image_done(image, seed, currentCount, upscaled=False):
        addQuery('txt2img', image, data, init_image, opt.strength, opt.iterations, opt.steps, opt.width, opt.height, opt.seamless, opt.fit, opt.mask, opt.invert_mask, opt.cfg_scale, opt.sampler_name, opt.gfpgan_strength, opt.upscale, opt.progress_images, opt.seed, opt.variation_amount, opt.with_variations)
        image.save(folderName + '/' + str(currentCount) + '.png', format='PNG')
        if (currentCount >= count):
            zipFileName = datetime.now().strftime("%Y%m%d%H%M%S") + 'images'
            shutil.make_archive(zipFileName, 'zip', folderName)
            for root, dirs, files in os.walk(folderName, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(folderName)
            _data = []
            with open(zipFileName + '.zip', 'rb') as f:
                _data = io.BytesIO(f.read())
            
            os.remove(zipFileName + '.zip')
            self.wfile.write(_data.read())
            return

        _model.prompt2imageWithCount(**vars(opt), step_callback=image_progress, image_callback=image_done, currentCount=currentCount)
    
    def image_progress(sample, step):
        pass

    try:
        init_image.save("./img2img-tmp.png")
        opt = build_opt2(data, strength, steps)
        opt.init_img = "./img2img-tmp.png"
        _model.prompt2imageWithCount(**vars(opt), step_callback=image_progress, image_callback=image_done, currentCount=0)
    finally:
        os.remove("./img2img-tmp.png")

def do_get(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    prompt = s
    model = get_model(id, self.address_string())
    renderImage(self, prompt, model)
    return

def do_get_image_mass(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    count = 1
    if 'count' in query_components:
        count = int(query_components['count'])

    prompt = s
    model = get_model(id, self.address_string())
    renderImageMass(self, prompt, model, count)
    return

def do_get_mod(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    color = 'FFFFFF'
    if "color" in query_components:
        color = query_components["color"]
    
    strength = 0.75
    if "strength" in query_components:
        strength = float(query_components["strength"])

    steps = 50
    if "steps" in query_components:
        steps = int(query_components["steps"])
    
    color_tuple = hex_color_string_to_tuple(color)
    init_image = create_img(512, 512, color_tuple)
    model = get_model(id, self.address_string())
    renderModImage(self, init_image, s, strength, steps, model)
    return

def do_get_mod_image_mass(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    color = 'FFFFFF'
    if "color" in query_components:
        color = query_components["color"]
    
    strength = 0.75
    if "strength" in query_components:
        strength = float(query_components["strength"])

    steps = 50
    if "steps" in query_components:
        steps = int(query_components["steps"])
    
    count = 1
    if 'count' in query_components:
        count = int(query_components['count'])
    
    color_tuple = hex_color_string_to_tuple(color)
    init_image = create_img(512, 512, color_tuple)
    model = get_model(id, self.address_string())
    renderModImageMass(self, init_image, s, strength, steps, model, count)
    return


def do_post_mod(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    strength = 0.75
    if "strength" in query_components:
        strength = float(query_components["strength"])

    steps = 50
    if "steps" in query_components:
        steps = int(query_components["steps"])
        
    content_length = int(self.headers['Content-Length'])
    post_data = json.loads(self.rfile.read(content_length))

    model = get_model(id, self.address_string())
    renderModImage(self, post_data['init_image'], s, strength, steps, model)

def do_post_mod_mass(self):
    query = urlparse(self.path).query
    query_components = dict(qc.split("=") for qc in query.split("&"))
    s = query_components["s"]
    id = default_model
    if "id" in query_components:
        id = query_components["id"]

    if s is None or s == "":
        self.send_response(404)
        self.end_headers()
        return
    
    strength = 0.75
    if "strength" in query_components:
        strength = float(query_components["strength"])

    steps = 50
    if "steps" in query_components:
        steps = int(query_components["steps"])
    
    count = 1
    if 'count' in query_components:
        count = int(query_components['count'])
        
    content_length = int(self.headers['Content-Length'])
    post_data = json.loads(self.rfile.read(content_length))

    model = get_model(id, self.address_string())
    renderModImageMass(self, post_data['init_image'], s, strength, steps, model, count)

def do_get_db(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()
    with open("./static/dream_web/db.html", "rb") as content:
        self.wfile.write(content.read())

def do_get_load_db(self):
    query = urlparse(self.path).query
    count = -1
    if query is not None and query != "":
        query_components = dict(qc.split("=") for qc in query.split("&"))
        if "count" in query_components:
            count = int(query_components["count"])

    data = getQueries(count)
    self.send_response(200)
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    self.wfile.write(bytes(json.dumps(data), 'utf-8'))