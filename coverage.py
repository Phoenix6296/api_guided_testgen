import os
import sys
import subprocess
from tqdm import tqdm
import signal
import json
import torch
import tensorflow as tf
import sklearn
import jax
import xgboost as xgb
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH=os.getenv('PRJ_ROOT_DIR') or os.getcwd()
YOUR_ID=os.getenv('YOUR_ID')


def get_source_root(lib):
    if lib == 'torch':
        return os.path.dirname(torch.__file__)
    if lib == 'tf':
        return os.path.dirname(tf.__file__)
    if lib == 'sklearn':
        return os.path.dirname(sklearn.__file__)
    if lib == 'jax':
        return os.path.dirname(jax.__file__)
    if lib == 'xgb':
        return os.path.dirname(xgb.__file__)
    return None

def get_coverage_json(lib, baseline, iter):
    os.environ["COVERAGE_FILE"] = f'.cov_{lib}_{baseline}'
    src = get_source_root(lib)
    path = f'{ROOT_PATH}/out/{iter}/exec/{baseline}/{lib}/'
    w_path = f'{ROOT_PATH}/out/{iter}/coverage/'
    os.makedirs(w_path, exist_ok=True)
    if not src:
        print(f'Unknown library for coverage source: {lib}')
        return
    if not os.path.exists(path):
        print(f'No exec path found for coverage: {path}')
        return
    for file in tqdm(sorted(os.listdir(path)), total=len(os.listdir(path)), ncols=70):
        cov_cmd = f'coverage run -a --source={src} {path+file}'
        try:
            subprocess.run(cov_cmd, shell=True, timeout=60)
            # subprocess.run("coverage erase", shell=True)
        except subprocess.TimeoutExpired:
            continue
    json_cmd = f'coverage json -o {w_path}/{lib}_{baseline}.json'
    subprocess.run(json_cmd, shell=True)

def parse_cov(lib, baseline, iter):
    path = f'out/{iter}/cov/{baseline}/{lib}/'
    total_covered = 0
    num_of_apis = 0
    class_covered = 0
    num_of_files = 0
    # walk all the API files
    for file in sorted(os.listdir(path)):
        if file.endswith('.json'):
            num_of_apis += 1
            print(file)
            with open(path+file) as f:
                cov = json.loads(f.read())
            package_1 = '/'.join(file.split('.')[1:-1])
            package_2 = '/'.join(file.split('.')[1:-2])
            total_covered += cov['totals']['percent_covered']
            pack_1 = False
            for cov_path, info in cov['files'].items():
                if package_1 in cov_path:
                    # print(info['summary'])
                    class_covered += info['summary']['percent_covered']
                    num_of_files += 1
                    pack_1 = True
            if not pack_1:
                for cov_path, info in cov['files'].items():
                    if package_2 in cov_path:
                        class_covered += info['summary']['percent_covered']
                        num_of_files += 1
            print('='*80)
    print(f'total: {total_covered} / {num_of_apis} = {total_covered/num_of_apis}')
    print(f'stmt: {class_covered} / {num_of_files} = {class_covered/num_of_files}')

if __name__=="__main__":
    lib = sys.argv[1]
    baseline =  sys.argv[2]
    iter = sys.argv[3]
    get_coverage_json(lib, baseline, iter)