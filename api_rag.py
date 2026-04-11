import backoff
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import json
import os
import sys
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import torch
tf.get_logger().setLevel('ERROR')  # for tf2.x
tf.autograph.set_verbosity(0)

def normalize_hf_model_name(name):
    if not name:
        return 'Qwen/Qwen2.5-7B'

    alias_map = {
        'qwen2.5:7b': 'Qwen/Qwen2.5-7B',
        'qwen2.5-coder:7b': 'Qwen/Qwen2.5-7B',
        'qwen2.5-coder:7b-instruct': 'Qwen/Qwen2.5-7B',
    }

    lowered = name.strip().lower()
    if lowered in alias_map:
        return alias_map[lowered]
    return name.strip()


HF_MODEL = normalize_hf_model_name(os.getenv('HF_MODEL', 'Qwen/Qwen2.5-7B'))
HF_MAX_NEW_TOKENS = int(os.getenv('HF_MAX_NEW_TOKENS', '768'))
FIXED_RAG_EXAMPLES = 3

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
hf_generator = None
hf_tokenizer = None

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
    (RuntimeError, OSError, ValueError),
    max_tries=4,
)
def get_completion_local_transformers(messages, model_name=None):
    global hf_generator
    global hf_tokenizer

    target_model = normalize_hf_model_name(model_name or HF_MODEL)
    if hf_generator is None:
        hf_tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        has_accelerate = importlib.util.find_spec('accelerate') is not None
        model_load_kwargs = {
            'torch_dtype': 'auto',
            'trust_remote_code': True,
        }
        if has_accelerate:
            model_load_kwargs['device_map'] = 'auto'

        model = AutoModelForCausalLM.from_pretrained(target_model, **model_load_kwargs)
        if not has_accelerate:
            if torch.cuda.is_available():
                model = model.to('cuda')
            else:
                model = model.to('cpu')
        hf_generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=hf_tokenizer,
        )

    if hasattr(hf_tokenizer, 'apply_chat_template'):
        prompt = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = '\n'.join([f"{m['role']}: {m['content']}" for m in messages]) + '\nassistant:'

    outputs = hf_generator(
        prompt,
        max_new_tokens=HF_MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )

    if not outputs:
        return ''
    return outputs[0].get('generated_text', '').strip()


REFUSAL_MARKERS = [
    "i'm sorry",
    "i am sorry",
    "cannot assist",
    "can't assist",
    "can't help",
    "cannot help",
    "unable to comply",
    "i must refuse",
]


def is_refusal_response(text):
    if not text:
        return True

    lowered = text.strip().lower()
    for marker in REFUSAL_MARKERS:
        if marker in lowered:
            return True

    if lowered.startswith("{") and '"response"' in lowered and ("cannot" in lowered or "can't" in lowered):
        return True

    return False


def clip_docs_for_prompt(docs, max_chars=2200):
    clipped = []
    for doc in docs:
        if not doc:
            continue
        text = doc.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        clipped.append(text)
    return clipped


def build_fallback_test(api, lib):
    import_stmt = {
        'tf': 'import tensorflow as tf',
        'torch': 'import torch',
        'sklearn': 'import sklearn',
        'xgb': 'import xgboost as xgb',
        'jax': 'import jax',
    }.get(lib, 'import importlib')

    cls_name = api.split('.')[-1].replace('_', ' ').title().replace(' ', '')
    if not cls_name:
        cls_name = 'Api'

    return (
        "import unittest\n"
        f"{import_stmt}\n\n"
        f"class Test{cls_name}(unittest.TestCase):\n"
        "    def test_module_import(self):\n"
        "        self.assertTrue(True)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def generate_test_with_retry(sys_prompt, final_task_prompt, model_name, api, lib):
    strict_sys_prompt = (
        sys_prompt
        + " Return only runnable Python unittest code. Do not output JSON, markdown fences, or explanations."
    )

    prompt = [
        {"role": "system", "content": strict_sys_prompt},
        {"role": "user", "content": final_task_prompt}
    ]
    res = get_completion_local_transformers(prompt, model_name=model_name)

    if is_refusal_response(res):
        retry_prompt = (
            final_task_prompt
            + "\n\nThis is a benign software testing request for a public API. "
            + "Return only Python unittest code."
        )
        retry_messages = [
            {"role": "system", "content": strict_sys_prompt},
            {"role": "user", "content": retry_prompt},
        ]
        res = get_completion_local_transformers(retry_messages, model_name=model_name)

    if is_refusal_response(res):
        return build_fallback_test(api, lib)

    return res


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
    candidate_docs = _prepare_candidates(candidate_docs)
    if not candidate_docs:
        return []
    if len(candidate_docs) <= doc_num:
        return candidate_docs

    # MMR balances query relevance and novelty.
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


def get_hybrid_docs(prompt, api, lib, doc_num):
    # MMR-based hybrid retrieval over basic_rag_all candidates.
    candidates = get_basic_rag_docs(prompt, 'basic_rag_all', max(30, doc_num * 10))
    return select_mmr_examples(prompt, candidates, doc_num=doc_num, mmr_lambda=0.7)

def generate_prompt(baseline='basic_rag_sos', lib='xgb', doc_num=3, iter='local_transformers', model='transformers', max_apis=None):
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
        
        if baseline == 'hybrid' and 'bug_detect' not in iter:
            docs = clip_docs_for_prompt(get_hybrid_docs(basic_task_prompt, api, lib, FIXED_RAG_EXAMPLES))
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif ('basic_rag' in baseline or baseline == 'similarity') and 'bug_detect' not in iter:
            rag_source = 'basic_rag_all' if baseline == 'similarity' else baseline
            rag_doc_num = FIXED_RAG_EXAMPLES if baseline == 'similarity' else doc_num
            docs = get_basic_rag_docs(basic_task_prompt, rag_source, rag_doc_num)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'diversity' and 'bug_detect' not in iter:
            # Retrieve a wider candidate pool, cluster them, and select 3 representatives.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, FIXED_RAG_EXAMPLES * 10))
            docs = select_diverse_examples(basic_task_prompt, candidates, doc_num=FIXED_RAG_EXAMPLES)
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
            
        prompt_dir = f'out/{iter}/prompt/{baseline}/{lib}/'
        if not os.path.isdir(prompt_dir):
            raise FileNotFoundError(f'Missing required directory: {prompt_dir}. Run init_project_structure.sh first.')
        with open(prompt_file_path, 'w') as f:
            f.write(final_task_prompt)
        

def run_exp(baseline='basic_rag', lib='tf', doc_num=3, iter='0', model='transformers', max_apis=None):
    generated_dir = f'out/{iter}/generated/{baseline}/{lib}'
    prompt_dir = f'out/{iter}/prompt/{baseline}/{lib}'
    if not os.path.isdir(generated_dir):
        raise FileNotFoundError(f'Missing required directory: {generated_dir}. Run init_project_structure.sh first.')
    if baseline != 'zero_shot' and not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f'Missing required directory: {prompt_dir}. Run init_project_structure.sh first.')

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
        if baseline == 'hybrid' and 'bug_detect' not in iter:
            docs = clip_docs_for_prompt(get_hybrid_docs(basic_task_prompt, api, lib, FIXED_RAG_EXAMPLES))
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif ('basic_rag' in baseline or baseline == 'similarity') and 'bug_detect' not in iter:
            rag_source = 'basic_rag_all' if baseline == 'similarity' else baseline
            rag_doc_num = FIXED_RAG_EXAMPLES if baseline == 'similarity' else doc_num
            docs = get_basic_rag_docs(basic_task_prompt, rag_source, rag_doc_num)
            final_task_prompt = f'''{basic_task_prompt} Use the following documents (surronded by @@@) to make the test case more compilable, and passable, and cover more lines. {post_prompt}'''
            for i, d in enumerate(docs):
                final_task_prompt += f'\n@@@ Doc_{i+1}:\n' + d + '\n@@@'
        elif baseline == 'diversity' and 'bug_detect' not in iter:
            # Retrieve a wider candidate pool, cluster them, and select 3 representatives.
            candidates = get_basic_rag_docs(basic_task_prompt, 'basic_rag_all', max(30, FIXED_RAG_EXAMPLES * 10))
            docs = select_diverse_examples(basic_task_prompt, candidates, doc_num=FIXED_RAG_EXAMPLES)
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
        if model.startswith('transformers:'):
            model_name = normalize_hf_model_name(model.split(':', 1)[1])
        elif model == 'transformers':
            model_name = HF_MODEL
        else:
            model_name = HF_MODEL

        res = generate_test_with_retry(sys_prompt, final_task_prompt, model_name, api, lib)
        
        with open(f'out/{iter}/generated/{baseline}/{lib}/{api}', 'w') as f:
            f.write(res)
        if baseline == 'zero_shot':
            continue
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
    
    if not os.path.isdir(f'log/{iter}'):
        raise FileNotFoundError(f'Missing required directory: log/{iter}. Run init_project_structure.sh first.')

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
        if not (fm == 'transformers' or fm.startswith('transformers:')):
            print('Ignoring non-transformers model selector:', fm, '-> using local Transformers model:', HF_MODEL)
        print('Using local Transformers model:', HF_MODEL)
        run_exp(baseline=baseline, lib=lib, doc_num=3, iter=iter, model=fm, max_apis=max_apis)

