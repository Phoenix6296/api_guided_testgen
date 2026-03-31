import backoff
from openai import OpenAI
import openai
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import json
import os
import sys
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # for tf2.x
tf.autograph.set_verbosity(0)

def normalize_ollama_base_url(raw_url: str) -> str:
    url = (raw_url or '').rstrip('/')
    if not url:
        url = 'http://10.5.30.32:11434'
    if not url.endswith('/v1'):
        url = f'{url}/v1'
    return url


OLLAMA_BASE_URL = normalize_ollama_base_url(os.getenv('OLLAMA_BASE_URL', 'http://10.5.30.32:11434'))
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '120'))

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=os.getenv('OLLAMA_API_KEY', 'ollama')
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def require_dir_exists(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'Required directory does not exist: {path}. '
            'Create it before running this script.'
        )

def get_covered(class_paths, cov_infos, api):
    # get search string depending on the case
    api_name = api.split('.')[-1]
    if api[0].isupper:
        search = f'class {api_name}('
    else:
        search = f'def {api_name}('
        
    covered_files = []
    for file, cov in cov_infos.items():
        if file in class_paths['paths'] and cov['summary']['percent_covered'] > 0:
            covered_file = {file: []}
            with open(file) as f:
                cov_file = f.readlines()
            with_cov_file = []
            api_line = 0
            api_lines = []
            api_end = len(cov_file)
            for i, line in enumerate(cov_file):
                # check the line number of target API
                # get all the lines that start defining func.
                if 'def ' in line:
                    api_lines.append(i)
                # get the line that starts defining target.
                if search in line:
                    api_line = i
                # annotate the covered and uncovered lines.
                if i in cov['executed_lines']:
                    annot_line = f'[COV] {line} [COV]'
                elif i in cov['missing_lines']:
                    annot_line = f'[UNC] {line} [UNC]'
                with_cov_file.append(annot_line)
            # get the end
            
            for i, line_num in enumerate(api_lines):
                if api_line == line_num:
                    if api_line != api_lines[-1]:
                        api_end = api_lines[i+1]
                    break
                
            with_cov_file = '\n'.join(with_cov_file[api_line:api_lines[api_end]]).strip()
            covered_file[file] = with_cov_file
            covered_files.append(covered_file)
    return covered_files
    
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException
    
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = embedding_model.encode(input)
        return batch_embeddings.tolist()

def load_doc(lib='tf', src='issues'):
    with open(f'data/api_db/{src}_{lib}.jsonl') as f:
        docs = [json.loads(x) for x in f.readlines()]
    return docs

@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIError,
    ),
    max_tries=4,
)
def get_completion_ollama(messages, model_name=None):
    target_model = model_name or OLLAMA_MODEL
    response = ollama_client.chat.completions.create(
        model=target_model,
        messages=messages,
        temperature=0,
        timeout=OLLAMA_TIMEOUT,
    )
    return response.choices[0].message.content.strip()


def get_api_list(lib='tf'):
    docs = load_doc(lib, 'apidoc')
    return [n['title'] for n in docs]

def load_api_doc(api, lib):
    with open(f'data/api_db/apidoc_{lib}.jsonl') as f:
        docs = [json.loads(x) for x in f.readlines()]
    for doc in docs:
        if doc['title'] == api:
            apidoc = doc
            break
    if isinstance(apidoc['nl_descs'], list):
        nl_descs = ' '.join(apidoc['nl_descs']).strip()
    else:
        nl_descs = apidoc['nl_descs']
    if isinstance(apidoc['ex_codes'], list):
        ex_codes = ' '.join(apidoc['ex_codes']).strip()
    else:
        ex_codes = apidoc['ex_codes']
    sig = apidoc['signature']
    api_str = f'Signature: {sig}\nDescriptions: {nl_descs}\nExample code: {ex_codes}'
    return api_str

def get_api_rag_docs(prompt, api, lib, src, doc_num):
    api_name = api.split('.')[-1]
    if api_name.startswith('__'):
        api_name = api_name[2:]
    embed_fn = MyEmbeddingFunction()
    client = chromadb.PersistentClient(path='./api_docs_db')
    
    collection = client.get_or_create_collection(
        name=f'{api_name}_{lib}_{src}',
        embedding_function=embed_fn
    )
    retriever_results = collection.query(
        query_texts=[prompt],
        n_results=doc_num,
    )
    doc = retriever_results["documents"][0]
    return doc
    

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
    doc = retriever_results["documents"][0]
    # doc = ' '.join(doc.split()[:200])
    return doc


def select_diverse_examples(query, candidate_docs, doc_num=3):
    if not candidate_docs:
        return []
    if len(candidate_docs) <= doc_num:
        return candidate_docs

    # Cluster similar candidates and choose one representative per cluster,
    # favoring the document with highest query similarity inside each cluster.
    texts = [query] + candidate_docs
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

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
    if not candidate_docs:
        return []
    if len(candidate_docs) <= doc_num:
        return candidate_docs

    # MMR: balance query relevance and novelty among already selected examples.
    texts = [query] + candidate_docs
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

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
        best_score = -1e9
        for idx in remaining:
            redundancy = float(np.max(np.dot(selected_embs, doc_embs[idx])))
            score = (mmr_lambda * relevance_scores[idx]) - ((1 - mmr_lambda) * redundancy)
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_docs[i] for i in selected]

def generate_prompt(baseline='basic_rag_sos', lib='xgb', doc_num=3, iter='local_ollama', model='ollama-small', max_apis=None):
    with open(f'data/api_db/api_class_over_10_{lib}.jsonl') as f:
        api_list = [json.loads(x)['api'] for x in f.readlines()]
    if max_apis is not None:
        api_list = api_list[:max_apis]
    
    library = {
        'tf': 'TensorFlow',
        'torch': 'PyTorch',
        'sklearn': 'Scikit-learn',
        'xgb': 'XGBoost',
        'jax': 'Google JAX'
    }
    library = library[lib]

    for api in tqdm(api_list, total=len(api_list), ncols=70):
        prompt_file_path = f'out/{iter}/prompt/{baseline}/{lib}/{api}'
        if os.path.exists(prompt_file_path):
            continue
            
        if 'bug_detect' in iter:
            if baseline == 'zero_shot':
                final_task_prompt = f'''Generate a python unit test cases to test the functionality of {api} API in {library} library. Your ultimate goal is to generate test cases that reveal bugs in this API.'''
            else:
                apidoc = get_basic_rag_docs(f'API doc for {api}', 'basic_rag_apidoc', 1)
                final_task_prompt = f'''Generate a python unit test cases to test the functionality of {api} API in {library} library. Your ultimate goal is to generate test cases that reveal bugs in this API. Use the following API documentation for reference: {apidoc}.'''
        else:
            basic_task_prompt = f'Generate a python unit test case to test the functionality of {api} API in {library} library with maximum coverage.'
        
        if 'limit' in iter:
            num = iter[-1]
            post_prompt = f'Generate a test suite with only {num} concise unit test(s). Use unittest library so it can be directly runnable (with the necessary imports and a main function).'
        else:
            post_prompt = 'Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).'
        
        if ('basic_rag' in baseline or baseline == 'similarity') and 'bug_detect' not in iter:
            rag_source = 'basic_rag_all' if baseline == 'similarity' else baseline
            docs = get_basic_rag_docs(basic_task_prompt, rag_source, doc_num)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'diversity' and 'bug_detect' not in iter:
            # Retrieve a wider candidate pool, cluster them, and select 3 representatives.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, doc_num * 10))
            docs = select_diverse_examples(basic_task_prompt, candidates, doc_num=3)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'hybrid' and 'bug_detect' not in iter:
            # Hybrid retrieval: dense retrieval + MMR re-ranking for relevance/diversity trade-off.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, doc_num * 10))
            docs = select_mmr_examples(basic_task_prompt, candidates, doc_num=3, mmr_lambda=0.7)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif 'api_rag' in baseline:
            if 'apidoc' in baseline:
                doc = load_api_doc(api, lib)
                final_task_prompt = f'''{basic_task_prompt} Use the following API documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}\n@@@ {doc} @@@'''
            elif 'issues' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'issues', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'sos' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'sos', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'repos' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'repos', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'all' in baseline:
                docs = []
                iss_doc = get_api_rag_docs(basic_task_prompt, api, lib, 'issues', 1)
                sos_doc = get_api_rag_docs(basic_task_prompt, api, lib, 'sos', 1)
                if iss_doc:
                    docs.append(iss_doc[0])
                if sos_doc:
                    docs.append(sos_doc[0])
                docs.append(load_api_doc(api, lib))
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'zero_shot' and 'bug_detect' not in iter:
            final_task_prompt = f'{basic_task_prompt} {post_prompt}'
        
        if baseline == 'zero_shot':
            continue
            
        require_dir_exists(f'out/{iter}/prompt/{baseline}/{lib}/')
        with open(prompt_file_path, 'w') as f:
            f.write(final_task_prompt)
        

def run_exp(baseline='basic_rag', lib='tf', doc_num=3, iter='0', model='ollama-small', max_apis=None):
    if baseline == 'iterative':
        with open(f'data/api_db/api_class_over_10_{lib}.jsonl') as f:
            classes = [x for x in f.readlines()]
    # if 'top_10' in iter:
    with open(f'data/api_db/api_class_over_10_{lib}.jsonl') as f:
        api_list = [json.loads(x)['api'] for x in f.readlines()]
    if max_apis is not None:
        api_list = api_list[:max_apis]
    # else:
    #     api_list = get_api_list(lib)
    library = {
        'tf': 'TensorFlow',
        'torch': 'PyTorch',
        'sklearn': 'Scikit-learn',
        'xgb': 'XGBoost',
        'jax': 'Google JAX'
    }
    library = library[lib]

    for api in tqdm(api_list, total=len(api_list), ncols=70) :
        if os.path.exists(f'out/{iter}/generated/{baseline}/{lib}/{api}'): continue
        if 'bug_detect' in iter:
            if baseline == 'zero_shot':
                final_task_prompt = f'''Generate a python unit test cases to test the functionality of {api} API in {library} library. Your ultimate goal is to generate test cases that reveal bugs in this API.'''
            else:
                apidoc = get_basic_rag_docs(f'API doc for {api}', 'basic_rag_apidoc', 1)
                final_task_prompt = f'''Generate a python unit test cases to test the functionality of {api} API in {library} library. Your ultimate goal is to generate test cases that reveal bugs in this API. Use the following API documentation for reference: {apidoc}.'''
        else:
            basic_task_prompt = f'Generate a python unit test case to test the functionality of {api} API in {library} library with maximum coverage.'
        if 'limit' in iter:
            num = iter[-1]
            post_prompt = f'Generate a test suite with only {num} concise unit test(s). Use unittest library so it can be directly runnable (with the necessary imports and a main function).'
        else:
            post_prompt = 'Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).'
        if ('basic_rag' in baseline or baseline == 'similarity') and 'bug_detect' not in iter:
            rag_source = 'basic_rag_all' if baseline == 'similarity' else baseline
            docs = get_basic_rag_docs(basic_task_prompt, rag_source, doc_num)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline in ('diversity') and 'bug_detect' not in iter:
            # Retrieve a wider candidate pool, cluster them, and select 3 representatives.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, doc_num * 10))
            docs = select_diverse_examples(basic_task_prompt, candidates, doc_num=3)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'hybrid' and 'bug_detect' not in iter:
            # Hybrid retrieval: dense retrieval + MMR re-ranking for relevance/diversity trade-off.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, doc_num * 10))
            docs = select_mmr_examples(basic_task_prompt, candidates, doc_num=3, mmr_lambda=0.7)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif 'api_rag' in baseline:
            if 'apidoc' in baseline:
                doc = load_api_doc(api, lib)
                final_task_prompt = f'''{basic_task_prompt} Use the following API documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}\n@@@ {doc} @@@'''
            elif 'issues' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'issues', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'sos' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'sos', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'repos' in baseline:
                docs = get_api_rag_docs(basic_task_prompt, api, lib, 'repos', 3)
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
            elif 'all' in baseline:
                docs = []
                iss_doc = get_api_rag_docs(basic_task_prompt, api, lib, 'issues', 1)
                sos_doc = get_api_rag_docs(basic_task_prompt, api, lib, 'sos', 1)
                if iss_doc:
                    docs.append(iss_doc[0])
                if sos_doc:
                    docs.append(sos_doc[0])
                docs.append(load_api_doc(api, lib))
                final_task_prompt = f'''{basic_task_prompt} Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
                for i, d in enumerate(docs):
                    final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'zero_shot' and 'bug_detect' not in iter:
            final_task_prompt = f'{basic_task_prompt} {post_prompt}'
        
        sys_prompt = f"You are a unit test suite generator for {library} library."
        if model.startswith('ollama:'):
            model_name = model.split(':', 1)[1]
        else:
            model_name = OLLAMA_MODEL

        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": final_task_prompt}
        ]
        res = get_completion_ollama(prompt, model_name=model_name)
        
        require_dir_exists(f'out/{iter}/generated/{baseline}/{lib}/')
        with open(f'out/{iter}/generated/{baseline}/{lib}/{api}', 'w') as f:
            f.write(res)
        if baseline == 'zero_shot':
            continue
        require_dir_exists(f'out/{iter}/prompt/{baseline}/{lib}/')
        with open(f'out/{iter}/prompt/{baseline}/{lib}/{api}', 'w') as f:
            f.write(final_task_prompt)
        

if __name__=="__main__":
    # tf/torch zero_shot/basic_rag 1/2/3/4/5
    lib = sys.argv[1]
    baseline = sys.argv[2]
    iter = sys.argv[3]
    fm = sys.argv[4]
    max_apis = None
    if len(sys.argv) >= 6:
        max_apis = int(sys.argv[5])
    
    require_dir_exists(f'log/{iter}')

    # generate_prompt(baseline=baseline, lib=lib, doc_num=3, iter=iter, model=fm)
    
    if baseline not in [
        'basic_rag_all',
        'basic_rag_apidoc',
        'basic_rag_sos',
        'basic_rag_issues',
        'basic_rag_repos',
        'similarity',
        'diversity',
        'hybrid',
        'zero_shot',
        'api_rag_all',
        'api_rag_apidoc',
        'api_rag_issues',
        'api_rag_sos',
        'api_rag_repos'
    ]:
        print('incorrect baseline!')
    else:
        print('Generating TCs for', lib, 'library using', baseline, 'iter', iter, 'model', fm)
        if not fm.startswith('ollama'):
            print('Ignoring non-local model selector:', fm, '-> using local Ollama model:', OLLAMA_MODEL)
        print('Using local Ollama endpoint:', OLLAMA_BASE_URL, 'model:', OLLAMA_MODEL)
        run_exp(baseline=baseline, lib=lib, doc_num=3, iter=iter, model=fm, max_apis=max_apis)

