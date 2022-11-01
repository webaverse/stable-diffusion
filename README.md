<h1 align='center'><b>Webaverse Stable Diffusion Toolkit</b></h1> 

Fork of InvokeAI's fork of Stable Diffusion by CompVis and RunwayML.

# Installation

This fork is supported across multiple platforms. You can find individual installation instructions below.

- ## [Linux](docs/installation/INSTALL_LINUX.md)
- ## [Windows](docs/installation/INSTALL_WINDOWS.md)
- ## [Macintosh](docs/installation/INSTALL_MAC.md)

# API Usage
To use the API:
```js
// text2img API
const params = { s: "a group of adventurers meet the lisk" }
const response = await fetch("https://stable-diffusion.webaverse.com/image",
{
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    method: "GET",
    params
})

// img2img API
const strength = 0.7; // how much should the original image be added to the noise?
const imageData; // optional imagedata
const params = { s: "a group of adventurers meet the lisk", strength }
const response = await fetch("https://stable-diffusion.webaverse.com/mod",
{
    method: "POST",
    body: imageData,
    params
})

// inpainting API
const strength = 0.7; // how much should the original image be added to the noise?
const inpaintingData = { image, mask }; // image data from canvases
const params = { s: "a group of adventurers meet the lisk", strength }
const response = await fetch("https://stable-diffusion.webaverse.com/inpaint",
{
    method: "POST",
    body: inpaintingData,
    params
})


    ```

## **Hardware Requirements**

**System**

You wil need one of the following:

- An NVIDIA-based graphics card with 4 GB or more VRAM memory.
- An Apple computer with an M1 chip.

**Memory**

- At least 12 GB Main Memory RAM.

**Disk**

- At least 6 GB of free disk space for the machine learning model, Python, and all its dependencies.

**Note**

If you are have a Nvidia 10xx series card (e.g. the 1080ti), please
run the dream script in full-precision mode as shown below.

Similarly, specify full-precision mode on Apple M1 hardware.

To run in full-precision mode, start `dream.py` with the
`--full_precision` flag:

```
(ldm) ~/stable-diffusion$ python scripts/dream.py --full_precision
```

# Features

## **Major Features**

- ## [Interactive Command Line Interface](docs/features/CLI.md)

- ## [Image To Image](docs/features/IMG2IMG.md)

- ## [Inpainting Support](docs/features/INPAINTING.md)

- ## [GFPGAN and Real-ESRGAN Support](docs/features/UPSCALE.md)

- ## [Seamless Tiling](docs/features/OTHER.md#seamless-tiling)

- ## [Google Colab](docs/features/OTHER.md#google-colab)

- ## [Web Server](docs/features/WEB.md)

- ## [Reading Prompts From File](docs/features/OTHER.md#reading-prompts-from-a-file)

- ## [Shortcut: Reusing Seeds](docs/features/OTHER.md#shortcuts-reusing-seeds)

- ## [Weighted Prompts](docs/features/OTHER.md#weighted-prompts)

- ## [Variations](docs/features/VARIATIONS.md)

- ## [Personalizing Text-to-Image Generation](docs/features/TEXTUAL_INVERSION.md)

- ## [Simplified API for text to image generation](docs/features/OTHER.md#simplified-api)

## **Other Features**

- ### [Creating Transparent Regions for Inpainting](docs/features/INPAINTING.md#creating-transparent-regions-for-inpainting)

- ### [Preload Models](docs/features/OTHER.md#preload-models)

# Latest Changes

- v1.14 (11 September 2022)

  - Memory optimizations for small-RAM cards. 512x512 now possible on 4 GB GPUs.
  - Full support for Apple hardware with M1 or M2 chips.
  - Add "seamless mode" for circular tiling of image. Generates beautiful effects. ([prixt](https://github.com/prixt)).
  - Inpainting support.
  - Improved web server GUI.
  - Lots of code and documentation cleanups.

- v1.13 (3 September 2022

  - Support image variations (see [VARIATIONS](docs/features/VARIATIONS.md) ([Kevin Gibbons](https://github.com/bakkot) and many contributors and reviewers)
  - Supports a Google Colab notebook for a standalone server running on Google hardware [Arturo Mendivil](https://github.com/artmen1516)
  - WebUI supports GFPGAN/ESRGAN facial reconstruction and upscaling [Kevin Gibbons](https://github.com/bakkot)
  - WebUI supports incremental display of in-progress images during generation [Kevin Gibbons](https://github.com/bakkot)
  - A new configuration file scheme that allows new models (including upcoming stable-diffusion-v1.5)
    to be added without altering the code. ([David Wager](https://github.com/maddavid12))
  - Can specify --grid on dream.py command line as the default.
  - Miscellaneous internal bug and stability fixes.
  - Works on M1 Apple hardware.
  - Multiple bug fixes.

For older changelogs, please visit **[CHANGELOGS](docs/CHANGELOG.md)**.

# Troubleshooting

Please check out our **[Q&A](docs/help/TROUBLESHOOT.md)** to get solutions for common installation problems and other issues.

# Contributing

Anyone who wishes to contribute to this project, whether documentation, features, bug fixes, code cleanup, testing, or code reviews, is very much encouraged to do so. If you are unfamiliar with
how to contribute to GitHub projects, here is a [Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github).

A full set of contribution guidelines, along with templates, are in progress, but for now the most important thing is to **make your pull request against the "development" branch**, and not against "main". This will help keep public breakage to a minimum and will allow you to propose more radical changes.

## **Contributors**

This fork is a combined effort of various people from across the world. [Check out the list of all these amazing people](docs/CONTRIBUTORS.md). We thank them for their time, hard work and effort.

# Support

For support,
please use this repository's GitHub Issues tracking service. Feel free
to send me an email if you use and like the script.

Original portions of the software are Copyright (c) 2020 Lincoln D. Stein (https://github.com/lstein)

# Further Reading

Please see the original README for more information on this software
and underlying algorithm, located in the file [README-CompViz.md](docs/README-CompViz.md).

# Add new models

To add new modesl go to configs/models.json and add the new model object in the array

```
  {
    "id": "stable-diffusion-1.4", //The id of the model that will be used to call it
    "name": "models/ldm/stable-diffusion-v1/model.ckpt", //The path for the model checkpoint
    "config": "configs/stable-diffusion/v1-inference.yaml", //The path for the model config
    "data": {
      "width": 512,
      "height": 512,
      "sampler_name": "k_lms",
      "full_precision": false,
      "grid": false,
      "seamless": false,
      "device_type": "cuda",
      "infile": null
    }
  }
```

# How to use the --web

The --web opens a webserver, that has a main ui, a database viewer and some headless routes
The main ui has options to generate images and shows the results.! (/)
![Screenshot_45](https://user-images.githubusercontent.com/45359358/193286613-519981a4-06e6-439d-bbfd-bf6d77c560ad.png)

The database viewer, can fetch old generations and their parameters and result (/db)
![Screenshot_42](https://user-images.githubusercontent.com/45359358/193286836-60ec6461-626b-4214-b1f6-88b07edecb65.png)

There are () other routes:
 * /image (get) 
   * arguments: s (prompt), id (model id, can be left out to use default model) 
   * generates an image
   * example: 127.0.0.1/image?s=cat&id=sd_v1.4
 * /image_mass (get) 
   * arguments: s (prompt), id (model id, can be left out to use default model), count (how many images to generate)
   * generates multiple images and returns a zip file
   * example: 127.0.0.1/image_mass?s=cat&id=sd_v1.4&count=4
 * /mod (get) 
   * arguments: s (prompt), id (model id, can be left out to use default model), color (the hex id for the colour to use, can be left to use white colour as default)
   * image to image generation
   * example: 127.0.0.1/mod?s=cat&id=sd_v1.4&color=03fce3
 * /mod_mass (get)
   * arguments: s (prompt), id (model id, can be left out to use default model), color (the hex id for the colour to use, can be left to use white colour as default), count (how many images to generate)
   * generates multiple image 2 image and returns a zip
   * example: 127.0.0.1/mod_mass?s=cat&id=sd_v1.4&color=03fce3&count=4
 * /mod (set)
   * arguments: s (prompt), id (model id, can be left out to use default model)
   * is used for image to image, it needs an image as the body request
 * /mod_mass(set)
   * arguments: s (prompt), id (model id, can be left out to use default model), count (how many generations it will do)
   * is used for image to image, it needs an image as the body request and returns a zip with all the generated images
 * /load_db (get)
   * arguments: count (can be left out, to get all data)
   * returns old generations (results & parameters)
