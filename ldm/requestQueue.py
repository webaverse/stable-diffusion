from redis import Redis
from ldm.env_reader import getValue, getNumber

r = None

def initQueue():
    global r
    r = Redis(host=getValue('REDIS_HOST'), port=getNumber('REDIS_PORT'), db=getNumber('REDIS_DB'), username=getValue('REDIS_USERNAME'), password=getValue('REDIS_PASSWORD'))
    setIsRunning('False')

def getIsRunning():
    if not r.exists('running'):
        return False

    res = r.get('running')
    if res == b'True':
        return True
    
    return False
    
def setIsRunning(val):
    if r == None:
        return

    r.set('running', val)