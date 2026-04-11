import os
import json
import ast
from tqdm import tqdm
import subprocess
import shutil
import sys

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.getenv('PRJ_ROOT_DIR') or os.getcwd()

def load_doc(lib='tf', src='issues'):
    # with open(f'{ROOT_PATH}/data/api_db/{src}_{lib}.jsonl') as f:
    with open(f'{ROOT_PATH}/data/api_db/api_class_over_10_{lib}.jsonl') as f:
        docs = [json.loads(x) for x in f.readlines()]
    return docs

def get_api_list(lib='tf'):
    docs = load_doc(lib, 'apidoc')
    return [n['api'] for n in docs]


def get_api_list_from_generated(lib, baseline, iter_name):
    generated_dir = f'{ROOT_PATH}/out/{iter_name}/generated/{baseline}/{lib}'
    if not os.path.exists(generated_dir):
        return []
    return sorted(
        [name for name in os.listdir(generated_dir) if os.path.isfile(f'{generated_dir}/{name}')]
    )

def parse_test_results(log):
    test_num = 0
    failures = {}
    # if execution success
    if 'Ran ' in log:
        log = log.split('Ran ')[-1]
        if 'tests' in log:
            test_num = int(log.split(' tests')[0])
        else:
            test_num = int(log.split(' test')[0])
        for line in log.splitlines():
            if 'FAILED' in line:
                fails = line.split('FAILED')[-1].replace('(', '').replace(')','').split(', ')
                for fail in fails:
                    fail = fail.split('=')
                    failures[fail[0].strip()] = int(fail[1].strip())
    # Execution failed
    else:
        test_num = -1

    return test_num, failures

def add_main(lib, code):
    if lib == 'torch':
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'tf':
        post = "\nif __name__ == '__main__':\n\ttf.test.main()"
    elif lib == 'sklearn':
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'jax':
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'xgb':
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    return code + post

def add_class(lib, code):
    if lib == 'torch':
        pre = 'import unittest\nimport torch\nimport numpy as np\nclass SmClass(unittest.TestCase):\n'
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'tf':
        pre = 'import tensorflow as tf\nimport numpy as np\n'
        post = '\nif __name__ == "__main__":\n\ttf.test.main()'
    elif lib == 'sklearn':
        pre = 'import sklearn\nimport numpy as np\nimport unittest\n'
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'jax':
        pre = 'import jax\nimport numpy as np\nimport unittest\n'
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    elif lib == 'xgb':
        pre = 'import xgboost as xgb\nimport numpy as np\nimport unittest\n'
        post = "\nif __name__ == '__main__':\n\tunittest.main()"
    new_code = ''
    for line in code.splitlines():
        new_code += f'\t{line}\n'
    return pre + new_code + post
        
        
def evaluate(lib='torch', baseline='basic_rag', iter='0', max_apis=None):
    api_list = []
    try:
        if 'top_10' in iter:
            with open(f'{ROOT_PATH}/data/api_db/api_class_over_10_{lib}.jsonl') as f:
                api_list = [json.loads(x)['api'] for x in f.readlines()]
        else:
            api_list = get_api_list(lib=lib)
    except FileNotFoundError:
        api_list = get_api_list_from_generated(lib=lib, baseline=baseline, iter_name=iter)

    if not api_list:
        print(f'No API list found for {lib}/{baseline}/{iter}.')
        return
    if max_apis is not None:
        api_list = api_list[:max_apis]
    parse_fail = 0
    total_tests = 0
    failures = 0
    errors = 0
    exec_cnt = 0
    path = f'{ROOT_PATH}/out/{iter}/exec/{baseline}/{lib}'
    # if os.path.exists(f'{ROOT_PATH}/log{iter}/{lib}_{baseline}_eval.log'):
    #     print('log already exists, so, skipping eval!')
    # #     return
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Missing required directory: {path}. Run init_project_structure.sh first.')
    for name in os.listdir(path):
        target = f'{path}/{name}'
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.isfile(target):
            os.remove(target)
    for i, api in enumerate(tqdm(api_list, total=len(api_list), ncols=70)):
        with open(f'{ROOT_PATH}/out/{iter}/generated/{baseline}/{lib}/{api}') as f:
            generated = f.read()
        if not generated:
            parse_fail += 1
            continue
        if '```python' in generated:
            # Extract content between the first ```python and closing ```
            generated = generated.split('```python', 1)[1].split('```', 1)[0].strip()
        print(api)
        exec_cnt += 1
        try:
            tree = ast.parse(generated)
        except SyntaxError:
            parse_fail += 1
            print('not parsable by ast')
            continue
        # Syntax Check: if not parsable by ast, count then skip
        if not tree:
            parse_fail += 1
            print('not parsable by ast')
        # Syntax Check: if it is parsable, execute it
        else:
            file = f'{path}/{api}.py'
            if 'import' not in generated:
                generated = add_class(lib, generated)
            elif 'import unittest' not in generated:
                generated = 'import unittest\n' + generated
            elif '__main__' not in generated:
                generated = add_main(lib, generated)
            with open(file, 'w') as f:
                f.write(generated)
            cmd = [sys.executable, file]
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60  # Set timeout as needed
                )
                std_err = result.stderr.decode("utf-8")
                test_num, fails = parse_test_results(std_err)
                # Execution Check: if execution succeeded, count the failing tests
                if test_num != -1:
                    total_tests += test_num
                    if 'failures' in fails:
                        failures += fails['failures']
                    if 'errors' in fails:
                        errors += fails['errors']
                print('test:', test_num)
                print('fails:', fails)
            # Handle timeout exception
            except subprocess.TimeoutExpired:
                total_tests += 1
                errors += 1
                print('skipped due to timeout')


    parse_rate = (len(api_list) - parse_fail) / len(api_list) * 100
    if total_tests == 0:
        error_rate = 0
        pass_rate = 0
    else:
        error_rate = ((total_tests - errors) / total_tests) * 100
        pass_rate = (total_tests - errors - failures) / total_tests * 100
    
    print(f'parse rate: {len(api_list) - parse_fail} / {len(api_list)} = {parse_rate}')
    print(f'exec rate: {total_tests - errors} / {total_tests} = {error_rate}')
    print(f'pass rate: {total_tests - errors - failures} / {total_tests} = {pass_rate}')
    print(f'.py count: {exec_cnt} / {len(api_list)} = {exec_cnt/len(api_list)}')
    
if __name__=="__main__":
    lib = sys.argv[1]       # tf/torch
    baseline = sys.argv[2]  # zero_shot/basic_rag/api_rag
    iter = sys.argv[3]
    max_apis = None
    if len(sys.argv) >= 5:
        max_apis = int(sys.argv[4])
    
    evaluate(lib=lib, baseline=baseline, iter=iter, max_apis=max_apis)