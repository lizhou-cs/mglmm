import io
import os
import cv2
import json
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset

LOAD_DATA_FROM_OSS = os.environ.get('LOAD_DATA_FROM_OSS')

OSS_ENV_KEYS = ("OSS_ACCESS_ID", "OSS_ACCESS_KEY", "OSS_BUCKET", "OSS_ENDPOINT")


def _load_oss2():
    try:
        import oss2
    except ImportError as exc:
        raise ImportError(
            "OSS support requires the optional dependency 'oss2'. "
            "Install it and set LOAD_DATA_FROM_OSS=1 only when loading data from Aliyun OSS."
        ) from exc
    return oss2


def _get_oss_config():
    missing_keys = [key for key in OSS_ENV_KEYS if not os.environ.get(key)]
    if missing_keys:
        raise RuntimeError(
            "Missing OSS environment variables: {}. "
            "Unset LOAD_DATA_FROM_OSS to use local files.".format(", ".join(missing_keys))
        )
    return {key: os.environ[key] for key in OSS_ENV_KEYS}


def _get_oss_bucket():
    oss2 = _load_oss2()
    oss_config = _get_oss_config()
    auth = oss2.Auth(oss_config['OSS_ACCESS_ID'], oss_config['OSS_ACCESS_KEY'])
    return oss2.Bucket(auth, oss_config['OSS_ENDPOINT'], oss_config['OSS_BUCKET'])

"""Load data from OSS"""
def _oss_get_file(file_path):
    return _get_oss_bucket().get_object(file_path)

def _oss_put_file(file_path, data):
    _get_oss_bucket().put_object(file_path, data)

def _oss_list_dir(dir_path, suffix=None):
    oss2 = _load_oss2()
    bucket = _get_oss_bucket()

    if suffix is None:
        return [obj.key for obj in oss2.ObjectIterator(bucket, prefix=dir_path)]
    else: 
        return [obj.key for obj in oss2.ObjectIterator(bucket, prefix=dir_path) if obj.key.endswith(suffix)]

def _oss_load_json(file_path):
    return json.load(_oss_get_file(file_path))

def _oss_load_img(img_path):
    try:
        img_bytes = _oss_get_file(img_path).read()
        img = Image.open(io.BytesIO(img_bytes))
        return img
    except:
        return None

class _OSSDataset(Dataset):
    MAX_RETRY = 5

    def __init__(self):
        self._oss2 = _load_oss2()
        self._bucket = _get_oss_bucket()

    def oss_get_file(self, file_path, retry=0):
        try:
            return self._bucket.get_object(file_path)
        except Exception as e:
            print(f"Failed to get file {file_path}: {e}")
            if retry < self.MAX_RETRY:
                print(f"Retrying {retry + 1}/{self.MAX_RETRY}...")
                return self.oss_get_file(file_path, retry=retry+1)
            else:
                print(f"Exceeded maximum retry limit for file {file_path}")
                return None
    
    def oss_list_dir(self, dir_path, suffix=None):
        if suffix is None:
            return [obj.key for obj in self._oss2.ObjectIterator(self._bucket, prefix=dir_path)]
        else: 
            return [obj.key for obj in self._oss2.ObjectIterator(self._bucket, prefix=dir_path) if obj.key.endswith(suffix)]
    
    def oss_load_json(self, file_path):
        return json.load(self.oss_get_file(file_path))
    
    def oss_load_img(self, img_path):
        try:
            img_bytes = self.oss_get_file(img_path).read()
            img = Image.open(io.BytesIO(img_bytes))
            return img
        except:
            print(f"Failed to load image {img_path}")
            return None
    
    def oss_load_img_cv2(self, img_path):
        try:
            img_bytes = self.oss_get_file(img_path).read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except:
            print(f"Failed to load image {img_path}")
            return None


"""Load data from Local"""
def _oss_get_file_local(file_path):
    return open(file_path, 'rb')

def _oss_put_file_local(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(data)

def _oss_list_dir_local(dir_path, suffix=None):
    if suffix is None:
        return [os.path.join(dir_path, obj) for obj in os.listdir(dir_path)]
    else: 
        return [os.path.join(dir_path, obj) for obj in os.listdir(dir_path) if obj.endswith(suffix)]

def _oss_load_json_local(file_path):
    return json.load(open(file_path, 'rb'))

def _oss_load_img_local(img_path):
    try:
        img = Image.open(img_path)
        return img
    except:
        print(f"Failed to load image {img_path}")
        return None


class _OSSDatasetLocal(Dataset):
    MAX_RETRY = 5

    def oss_get_file(self, file_path, retry=0):
        try:
            return open(file_path, 'rb')
        except Exception as e:
            print(f"Failed to get file {file_path}: {e}")
            if retry < self.MAX_RETRY:
                print(f"Retrying {retry + 1}/{self.MAX_RETRY}...")
                return self.oss_get_file(file_path, retry=retry+1)
            else:
                print(f"Exceeded maximum retry limit for file {file_path}")
                return None
    
    def oss_list_dir(self, dir_path, suffix=None):
        if suffix is None:
            return [os.path.join(dir_path, obj) for obj in os.listdir(dir_path)]
        else: 
            return [os.path.join(dir_path, obj) for obj in os.listdir(dir_path) if obj.endswith(suffix)]
    
    def oss_load_json(self, file_path):
        return json.load(self.oss_get_file(file_path))
    
    def oss_load_img(self, img_path, retry=0):
        try:
            img = Image.open(img_path)
            return img
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            if retry < self.MAX_RETRY:
                print(f"Retrying {retry + 1}/{self.MAX_RETRY}...")
                return self.oss_load_img(img_path, retry=retry+1)
            else:
                print(f"Exceeded maximum retry limit for image {img_path}")
                return None
    
    def oss_load_img_cv2(self, img_path, retry=0):
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            print(f"Failed to load image {img_path}: {e}")
            if retry < self.MAX_RETRY:
                print(f"Retrying {retry + 1}/{self.MAX_RETRY}...")
                return self.oss_load_img_cv2(img_path, retry=retry+1)
            else:
                print(f"Exceeded maximum retry limit for image {img_path}")
                return None


if LOAD_DATA_FROM_OSS is not None:
    OSSDataset = _OSSDataset
    oss_get_file = _oss_get_file
    oss_put_file = _oss_put_file
    oss_list_dir = _oss_list_dir
    oss_load_json = _oss_load_json
    oss_load_img = _oss_load_img
else:
    OSSDataset = _OSSDatasetLocal
    oss_get_file = _oss_get_file_local
    oss_put_file = _oss_put_file_local
    oss_list_dir = _oss_list_dir_local
    oss_load_json = _oss_load_json_local
    oss_load_img = _oss_load_img_local
