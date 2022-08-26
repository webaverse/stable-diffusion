import argparse, os, sys, glob
from argparse import Namespace
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import io
import copy
from flask import Flask, request, send_file, make_response, abort

app = Flask(__name__)

def mkResponse(data):
  return send_file(
    data,
    download_name="image.png",
    mimetype="image/png",
  )

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

#
# arguments
#

parser = argparse.ArgumentParser()

# txt2img
parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=2,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=3,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

opt = parser.parse_args([
    '--prompt',
    'anime girl',
    '--plms',
    '--ckpt',
    'stable-diffusion-v-1-3/sd-v1-3-full-ema.ckpt',
    '--n_samples',
    '1',
    '--skip_grid',
])
opt = vars(opt);

# img2img
parser2 = argparse.ArgumentParser()

parser2.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)

parser2.add_argument(
    "--init-img",
    type=str,
    nargs="?",
    help="path to the input image"
)

parser2.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/img2img-samples"
)

parser2.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)

parser2.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save indiviual samples. For speed measurements.",
)

parser2.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser2.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser2.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across all samples ",
)

parser2.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser2.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser2.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser2.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor, most often 8 or 16",
)
parser2.add_argument(
    "--n_samples",
    type=int,
    default=2,
    help="how many samples to produce for each given prompt. A.k.a batch size",
)
parser2.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser2.add_argument(
    "--scale",
    type=float,
    default=5.0,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser2.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser2.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser2.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser2.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser2.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser2.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

opt2 = parser2.parse_args([
    '--prompt',
    'anime girl',
    "--strength",
    "0.8",
    '--ckpt',
    'stable-diffusion-v-1-3/sd-v1-3-full-ema.ckpt',
    '--n_samples',
    '1',
    '--skip_grid',
])
opt2 = vars(opt2)

#
# initialize
#

# txt2img

seed_everything(opt.seed)

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# os.makedirs(opt.outdir, exist_ok=True)
# outpath = opt.outdir

batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

#
# methods
#

def load_img(postData):
    image = Image.open(io.BytesIO(postData)).convert('RGB')
    return image

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

def renderImage(data, localOpt):
    if localOpt.plms:
        samplerImage = PLMSSampler(model)
    else:
        samplerImage = DDIMSampler(model)

    precision_scope = autocast if localOpt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(localOpt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if localOpt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [localOpt.C, localOpt.H // localOpt.f, localOpt.W // localOpt.f]
                        samples_ddim, _ = samplerImage.sample(S=localOpt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=localOpt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=localOpt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=localOpt.ddim_eta,
                                                        x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not localOpt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # Image.fromarray(x_sample.astype(np.uint8)).save(
                                #     os.path.join(sample_path, f"{base_count:05}.png"))
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # base_count += 1

                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr.seek(0)

                                response = mkResponse(img_byte_arr)

                                response.headers["Access-Control-Allow-Origin"] = "*"
                                response.headers["Access-Control-Allow-Headers"] = "*"
                                response.headers["Access-Control-Allow-Methods"] = "*"
                                return response

    # print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
    #     f" \nEnjoy.")
    response = make_response("no result", 500)
    return response

def renderMod(data, init_image, localOpt):
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    if localOpt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        samplerMod = PLMSSampler(model)
    else:
        samplerMod = DDIMSampler(model)
    samplerMod.make_schedule(ddim_num_steps=localOpt.ddim_steps, ddim_eta=localOpt.ddim_eta, verbose=False)

    assert 0. <= localOpt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(localOpt.strength * localOpt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if localOpt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(localOpt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if localOpt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = samplerMod.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = samplerMod.decode(z_enc, c, t_enc, unconditional_guidance_scale=localOpt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not localOpt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # Image.fromarray(x_sample.astype(np.uint8)).save(
                                #     os.path.join(sample_path, f"{base_count:05}.png"))
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # base_count += 1

                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr.seek(0)

                                response = mkResponse(img_byte_arr)

                                response.headers["Access-Control-Allow-Origin"] = "*"
                                response.headers["Access-Control-Allow-Headers"] = "*"
                                response.headers["Access-Control-Allow-Methods"] = "*"
                                return response

    # print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
    #       f" \nEnjoy.")
    response = make_response("no result", 500)
    return response
    
#
# request handlers
#

@app.route("/image", methods=["GET", "OPTIONS"])
def image():
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        return response

    s = request.args.get("s")
    response = None
    if s is None or s == "":
        response = make_response("no text provided", 400)
    else:
        data = None
        if not opt.from_file:
            prompt = s
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        localOpt = copy.deepcopy(opt)
        localOpt.ddim_steps = request.args.get("n", default=localOpt.ddim_steps, type=int)
        localOpt.W = request.args.get("w", default=localOpt.W, type=int)
        localOpt.H = request.args.get("h", default=localOpt.H, type=int)
        return renderImage(data, localOpt)

def hex_color_string_to_tuple(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@app.route("/mod", methods=["GET", "POST", "OPTIONS"])
def reimage():
    if request.method == "OPTIONS":
        response = make_response("", 200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        return response

    s = request.args.get("s")
    response = None
    if s is None or s == "":
        response = make_response("no text provided", 400)
    else:
        data = None
        if not opt.from_file:
            prompt = s
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        localOpt = copy.deepcopy(opt2)
        localOpt.ddim_steps = request.args.get("n", default=localOpt.ddim_steps, type=int)
        localOpt.strength = request.args.get("noise", default=localOpt.strength, type=float)
        localOpt.W = request.args.get("w", default=localOpt.W, type=int)
        localOpt.H = request.args.get("h", default=localOpt.H, type=int)
        init_image = None
        if request.method == "GET":
            color_hex = None
            if request.args.get("color") is not None:
                color_hex = request.args.get("color")
            else:
                color_hex = "FFFFFF"
            color_tuple = hex_color_string_to_tuple(color_hex)
            init_image = create_img(512, 512, color_tuple)
        else:
            postData = request.get_data()
            init_image = load_img(postData)
        init_image = convert_img(init_image).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        return renderMod(data, init_image, localOpt)

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    print("Starting server...")
    app.run(
        host="0.0.0.0",
        port=80,
        debug=False,
        # dev_tools_silence_routes_logging = False,
        # dev_tools_ui=True,
        # dev_tools_hot_reload=True,
        threaded=True,
    )