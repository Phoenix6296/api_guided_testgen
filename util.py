import os 
import sys
import json
import ast
import inspect
import numpy as np
from tqdm import tqdm 
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import chromadb
import torch
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()
YOUR_ID=os.getenv('YOUR_ID')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def baseline_dir_name(baseline):
    aliases = {
        'similarity': 'similarity',
    }
    return aliases.get(baseline, baseline)

def resolve_baseline_dir(iter_name, baseline, bucket, lib):
    preferred = baseline_dir_name(baseline)
    preferred_path = f'out/{iter_name}/{bucket}/{preferred}/{lib}'
    if os.path.exists(preferred_path):
        return preferred
    fallback_path = f'out/{iter_name}/{bucket}/{baseline}/{lib}'
    if os.path.exists(fallback_path):
        return baseline
    return preferred

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = embedding_model.encode(input)
        return batch_embeddings.tolist()

def get_basic_rag_docs(prompt, src, doc_num):
    embed_fn = MyEmbeddingFunction()
    client = chromadb.PersistentClient(path='./docs_db')
    
    collection = client.get_or_create_collection(
        name=src,
        embedding_function=embed_fn
    )
    retriever_results = collection.query(
        query_texts=[prompt],
        n_results=doc_num,
    )
    return retriever_results["documents"][0]


def _prepare_candidates(candidate_docs):
    cleaned = []
    seen = set()
    for doc in candidate_docs:
        if not doc:
            continue
        text = doc.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def select_diverse_examples(query, candidate_docs, doc_num=3):
    candidate_docs = _prepare_candidates(candidate_docs)
    if not candidate_docs:
        return []
    if len(candidate_docs) <= doc_num:
        return candidate_docs

    texts = [query] + candidate_docs
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    doc_embs = np.array(embeddings[1:])

    n_clusters = min(doc_num, len(candidate_docs))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(doc_embs)

    relevance_scores = np.dot(doc_embs, query_emb)
    selected_indices = []
    for cluster_id in range(n_clusters):
        cluster_members = np.where(labels == cluster_id)[0]
        best_idx = max(cluster_members, key=lambda i: relevance_scores[i])
        selected_indices.append(int(best_idx))

    selected_indices = sorted(selected_indices, key=lambda i: relevance_scores[i], reverse=True)
    return [candidate_docs[i] for i in selected_indices[:doc_num]]


def select_mmr_examples(query, candidate_docs, doc_num=3, mmr_lambda=0.7):
    candidate_docs = _prepare_candidates(candidate_docs)
    if not candidate_docs:
        return []
    if len(candidate_docs) <= doc_num:
        return candidate_docs

    texts = [query] + candidate_docs
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    doc_embs = np.array(embeddings[1:])

    relevance_scores = np.dot(doc_embs, query_emb)
    selected = []
    remaining = list(range(len(candidate_docs)))

    while remaining and len(selected) < doc_num:
        if not selected:
            best_idx = max(remaining, key=lambda i: relevance_scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)
            continue

        selected_embs = doc_embs[selected]
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            redundancy = float(np.max(np.dot(selected_embs, doc_embs[idx])))
            score = (mmr_lambda * relevance_scores[idx]) - ((1 - mmr_lambda) * redundancy)
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_docs[i] for i in selected]

def get_api_list(lib='tf'):
    try:
        docs = load_doc(lib, 'apidoc')
        return [n['title'] for n in docs]
    except FileNotFoundError:
        fallback_path = f'data/api_db/api_class_over_10_{lib}.jsonl'
        if os.path.exists(fallback_path):
            with open(fallback_path) as f:
                docs = [json.loads(x) for x in f.readlines()]
            # Supports both {"api": "..."} and {"title": "..."} shapes.
            return [d.get('api', d.get('title')) for d in docs if d.get('api') or d.get('title')]
        return []

def load_doc(lib='tf', src='issues'):
    with open(f'data/api_db/{src}_{lib}.jsonl') as f:
        docs = [json.loads(x) for x in f.readlines()]
    return docs

def parse_cov(lib, baseline, iter):
    baseline_dir = baseline_dir_name(baseline)
    path = f'out/{iter}/cov/{baseline_dir}/{lib}/'
    
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
        
def count_avg_tests(lib, baseline, iter):
    baseline_dir = resolve_baseline_dir(iter, baseline, 'generated', lib)
    if 'top_10' in iter:
        docs = load_doc(lib, 'sorted')
        api_list = [n['api_name'] for n in docs]
        api_list = api_list[:round(len(api_list)/10)]
    else:
        api_list = get_api_list(lib)
    all_cnts = []
    for api in api_list:
        with open(f'out/{iter}/generated/{baseline_dir}/{lib}/{api}') as f:
            generated = f.read()
        cnt = 0
        for tok in generated.split():
            if tok == 'def':
                cnt += 1
        all_cnts.append(cnt)

    if not all_cnts:
        print('avg tests: 0')
        print('total tests: 0')
        return
        
    print('avg tests:', sum(all_cnts)/len(all_cnts))
    print('total tests:', sum(all_cnts))
    
def get_package(lib, iter):
    api_list = get_api_list(lib)
    
    for api in api_list:
        with open(f'out/{iter}/exec/basic_rag/{lib}/{api}.py') as f:
            code = f.readlines()
        print(api)
        for lin in code:
            if api in lin:
                print(lin)
        return
    
def parse_eval_log(lib, baseline, iter):
    with open(f'log/{iter}/{lib}_{baseline}_eval.log') as f:
        log = f.readlines()
    
    scores = []
    for line in log:
        line = line.strip()
        if line.startswith(lib):
            tmp = {'api': line}
        elif line.startswith('test:'):
            test = line.split()[-1]
            tmp['test'] = int(test)
        elif line.startswith('fails:'):
            fails = line.split('fails: ')[-1].replace('{', '').replace('}','').split(',')
            # print(fails)
            failures = {}
            for fail in fails:
                fail = fail.replace("'", '').strip()
                fail = fail.split(': ')
                if len(fail) > 1:
                    # print(fail)
                    failures[fail[0]] = int(fail[1]) 
            tmp['failures'] = failures
            scores.append(tmp)
        elif 'skipped due to timeout' in line:
            skipped = {
                'api': 'sm_test',
                'test': 1,
                'failures': {'errors': 1}
            }
            scores.append(skipped)
    
    test_num = 0
    fail = 0
    execute = 0
    parse = 0
    error = 0
    for scr in scores:
        print(scr)
        
        if scr['test'] > 0:
            test_num += scr['test']
        else:
            parse += 1
        if 'failures' in scr['failures'].keys():
            fail += scr['failures']['failures']
        if 'errors' in scr['failures'].keys():
            execute += 1
            error += scr['failures']['errors']

    print(f'parse rate: {len(scores) - parse} / {len(scores)} = {(len(scores) - parse)/len(scores)}')
    print(f'exec rate: {len(scores) - execute} / {len(scores)} = {(len(scores) - execute) / len(scores)}')
    print(f'error rate: {error} / {test_num} = {error / test_num}')
    print(f'pass rate: {test_num - error - fail} / {test_num} = {(test_num - error - fail) / test_num}')
    
def get_api_class(api, src):
    api_name = api.split('.')[-1]

    if api_name[0].isupper():
        search = f'class {api_name}('
    else:
        search = f'def {api_name}('
    paths = []
    for root, dirs, files in sorted(os.walk(src)):
        for file in files:
            if file.endswith('.py'):
                with open(f'{root}/{file}', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                if search in code:
                    paths.append(f'{root}/{file}')
    return paths
    
def write_api_class(lib):
    if lib == 'torch':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/torch'
    elif lib == 'tf':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/tensorflow'
    with open(f'data/api_db/apidoc_{lib}.jsonl') as f:
        apis = [json.loads(r) for r in f.readlines()]
    
    with open(f'data/api_db/api_class_{lib}.jsonl', 'w') as f: 
        for api in tqdm(apis, total=len(apis), ncols=75):
            paths = get_api_class(api['title'], src)
            w_dict = {'api': api['title'],
                    'paths': paths}
            f.write(json.dumps(w_dict) + '\n')
            # return
            
def write_api_class_top(lib):
    if lib == 'torch':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/torch'
    elif lib == 'tf':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/tensorflow'
    elif lib == 'sklearn':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/sklearn'
    elif lib == 'jax':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/jax'
    elif lib == 'xgb':
        src = f'/home/{YOUR_ID}/anaconda3/lib/python3.8/site-packages/xgboost'
        
    with open(f'data/api_db/sorted_{lib}_over10.jsonl') as f:
        apis = [json.loads(r)['api_name'] for r in f.readlines()]
    # t_apis = apis[:round(len(apis)/10)]
    # if len(t_apis) < 10:
    #     t_apis = apis[:25]
        
    with open(f'data/api_db/api_class_over_10_{lib}.jsonl', 'w') as f: 
        for api in apis:
            paths = get_api_class(api, src)
            w_dict = {'api': api,
                    'paths': paths}
            f.write(json.dumps(w_dict) + '\n')
        
        
def get_class_coverage(lib, baseline, iter):
    baseline_dir = baseline_dir_name(baseline)
    cov_path = f'out/{iter}/coverage/{lib}_{baseline_dir}.json'
    if not os.path.exists(cov_path):
        print(f'class cov: skipped (missing coverage file: {cov_path})')
        return
    with open(cov_path) as f:
        cov_infos = json.loads(f.read())

    with open(f'data/api_db/api_class_over_10_{lib}.jsonl') as f:
        paths = [json.loads(r) for r in f.readlines()]

    has_path_map = any('paths' in p for p in paths)
    coverages = []
    unique = []
    for file, cov in cov_infos.get('files', {}).items():
        if has_path_map:
            for path in paths:
                file_paths = path.get('paths', [])
                if file in file_paths and file not in unique:
                    unique.append(file)
                    coverages.append(cov['summary']['percent_covered'])
        else:
            # Fallback when api_class_over_10_{lib}.jsonl does not include "paths".
            coverages.append(cov['summary']['percent_covered'])

    if not coverages:
        print('class cov: 0')
        return
    print('class cov:', sum(coverages) / len(coverages))
    
    
def calculate_cost(lib, baseline, iter):
    baseline_dir = resolve_baseline_dir(iter, baseline, 'generated', lib)
    library = {
        'tf': 'TensorFlow',
        'torch': 'PyTorch',
        'sklearn': 'Scikit-learn',
        'xgb': 'XGBoost',
        'jax': 'Google JAX'
    }
    out_lengths = []
    for cur_dir, _, files in sorted(os.walk(f'out/{iter}/generated/{baseline_dir}/{lib}')):
        for file in sorted(files):
            with open(f'{cur_dir}/{file}') as f:
                generated = f.read()
            token_len = len(generated.split())
            out_lengths.append(token_len)
    
    in_lengths = []
    api_list = get_api_list(lib)
    for api in api_list:
        doc_prompt = f'Generate a python unit test case to test the functionality of {api} API in {library.get(lib)} library with maximum coverage.'
        basic_prompt = 'Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).'
        gpt_prompt = f'{doc_prompt} {basic_prompt}'
        if 'basic_rag' in baseline_dir:
            docs = get_basic_rag_docs(doc_prompt, baseline_dir, 3)
            gpt_prompt = f'''{doc_prompt} Use the following documents (surronded by @@@) to make the test case more compilable and passable. {basic_prompt}'''
            for i, d in enumerate(docs):
                gpt_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline_dir == 'similarity':
            docs = get_basic_rag_docs(doc_prompt, 'basic_rag_all', 3)
            gpt_prompt = f'''{doc_prompt} Use the following documents (surronded by @@@) to make the test case more compilable and passable. {basic_prompt}'''
            for i, d in enumerate(docs):
                gpt_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline_dir == 'diversity':
            candidates = get_basic_rag_docs(doc_prompt, 'basic_rag_all', 30)
            docs = select_diverse_examples(doc_prompt, candidates, doc_num=3)
            gpt_prompt = f'''{doc_prompt} Use the following documents (surronded by @@@) to make the test case more compilable and passable. {basic_prompt}'''
            for i, d in enumerate(docs):
                gpt_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline_dir == 'hybrid':
            candidates = get_basic_rag_docs(doc_prompt, 'basic_rag_all', 30)
            docs = select_mmr_examples(doc_prompt, candidates, doc_num=3, mmr_lambda=0.7)
            gpt_prompt = f'''{doc_prompt} Use the following documents (surronded by @@@) to make the test case more compilable and passable. {basic_prompt}'''
            for i, d in enumerate(docs):
                gpt_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline_dir == 'zero_shot':
            gpt_prompt = f'{doc_prompt} {basic_prompt}'
        token_len = len(gpt_prompt.split())
        in_lengths.append(token_len)

    if not in_lengths:
        print('input tokens: 0')
    else:
        print(f'input tokens: {sum(in_lengths) / len(in_lengths)}')

    if not out_lengths:
        print('output tokens: 0')
    else:
        print(f'output tokens: {sum(out_lengths) / len(out_lengths)}')
    
if __name__=="__main__": 
    if len(sys.argv) > 1:
        lib = sys.argv[1]
    if len(sys.argv) > 2:
        baseline = sys.argv[2]
    if len(sys.argv) > 3:
        iter = sys.argv[3]
        
    # get_package(lib)
    # parse_cov(lib, baseline)
    # parse_eval_log(lib, baseline)
    # write_api_class(lib)
    # write_api_class_top(lib)
    
    print(f'========= For {lib} {baseline} {iter} =========')
    get_class_coverage(lib, baseline, iter)
    calculate_cost(lib, baseline, iter)
    count_avg_tests(lib, baseline, iter)