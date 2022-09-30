import argparse
from contextlib import nullcontext
from urllib.parse import urlparse
from ldm.dream.model_leader import get_model
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torch import autocast
import torch
from tqdm import tqdm, trange
from einops import rearrange
from PIL import Image
import numpy as np
import io
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

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

def makeOpt():
    return parser.parse_args([
        '--prompt',
        'anime girl',
        '--plms',
        '--n_samples',
        '1',
        '--skip_grid',
    ])

def renderImage(self, data, localOpt, model_id):
    _model = get_model(model_id, False)
    batch_size = localOpt.n_samples
    start_code = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if localOpt.fixed_code:
        start_code = torch.randn([localOpt.n_samples, localOpt.C, localOpt.H // localOpt.f, localOpt.W // localOpt.f], device=device)
    if localOpt.plms:
        samplerImage = PLMSSampler(_model)
    else:
        samplerImage = DDIMSampler(_model)

    precision_scope = autocast if localOpt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with _model.ema_scope():
                for n in trange(localOpt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if localOpt.scale != 1.0:
                            uc = _model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = _model.get_learned_conditioning(prompts)
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

                        x_samples_ddim = _model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not localOpt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))

                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr.seek(0)

                                self.send_response(200)
                                self.send_header("Access-Control-Allow-Origin", "*")
                                self.send_header('Content-type', 'image/png')
                                self.end_headers()
                                self.wfile.write(img_byte_arr.read())

    self.send_response(404)
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Content-type", "application/json")
    self.end_headers()
    self.wfile.write(bytes('{}', 'utf8'))

default_model = "diffusion-1.4"

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
    localOpt = makeOpt()

    renderImage(self, prompt, localOpt, id)
    return