import json
from ldm.generate import Generate
from threading import RLock

lock = RLock()
models = []
last_loaded_model = {}

def load_models(path):
    global models
    f = open(path, "r", encoding="utf-8")
    _data = f.read()
    data = json.loads(_data)
    f.close()
    models = data

def get_model(id, client_address = '127.0.0.1'):
    with lock:
        if client_address in last_loaded_model:
            if last_loaded_model[client_address]['id'] == id and last_loaded_model[client_address]['model'] != None:
                return last_loaded_model[client_address]['model']

            for d in models:
                if d['id'] == id:
                    if (last_loaded_model[client_address]['model'] != None):
                        del last_loaded_model[client_address]['model']

                    data = d['data']
                    t2i = Generate(
                    width=data['width'],
                    height=data['height'],
                    sampler_name=data['sampler_name'],
                    weights=d['name'],
                    full_precision=data['full_precision'],
                    config=d['config'],
                    grid=data['grid'],
                    seamless=data['seamless'],
                    embedding_path=None,
                    device_type=data['device_type'],
                    ignore_ctrl_c=data['infile'] is None,
                    )
                    t2i.load_model()
                    last_loaded_model[client_address]['id'] = id
                    last_loaded_model[client_address]['model'] = t2i
                    return t2i
        else:
            last_loaded_model[client_address] = { 'id': None, 'model': None }
            return get_model(id, client_address)

    return None
