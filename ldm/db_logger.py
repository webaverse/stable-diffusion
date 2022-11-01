import sqlite3
import os
import io
from datetime import datetime
from PIL import Image
import codecs
import base64

db_con = None
cur = None

def initDbConnection(db_name = 'db.sqlite3'):
    exists = os.path.exists('db.sqlite3')
    global db_con, cur

    db_con = sqlite3.connect(db_name, check_same_thread=False)
    cur = db_con.cursor()
    print('connected to database')

    if not exists:
        cur.execute("CREATE TABLE queries(type, image, prompt, init_image, strength, iterations, steps, width, height, seamless, fit, mask, invert_mask, cfg_scale, sampler_name, gfpgan_strength, upscale, progress_images, seed, variation_amount, with_variations, datetime)")
    
def addQuery(type, image, prompt, init_image, strength, iterations, steps, width, height, seamless, fit, mask, invert_mask, cfg_scale, sampler_name, gfpgan_strength, upscale, progress_images, seed, variation_amount, with_variations):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    init_img_str = ""
    if init_image is not None:    
        buffered = io.BytesIO()
        init_image.save(buffered, format="PNG")
        init_img_str = base64.b64encode(buffered.getvalue()).decode()
    
    cur.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (type, img_str, prompt, init_img_str, strength, iterations, steps, width, height, seamless, fit, mask, invert_mask, cfg_scale, sampler_name, gfpgan_strength, upscale, progress_images, seed, variation_amount, with_variations, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    db_con.commit()

def getQueries(count):
    cur.execute("SELECT * FROM queries")
    data = cur.fetchall()
    i = 0
    if (count < 0):
        count = len(data)

    sortedArray = sorted(data, key=lambda t: datetime.strptime(t[21], '%Y-%m-%d %H:%M:%S'), reverse=True)
    res = []

    while i < len(sortedArray):
        res.append(sortedArray[i])
        i += 1
        if i >= count:
            break
        
    return res