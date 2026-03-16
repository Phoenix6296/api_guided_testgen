import json
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import sys
from dotenv import load_dotenv
load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings = embedding_model.encode(input)
        return batch_embeddings.tolist()

def load_basic_doc(src):
    with open(f'data/api_db/{src}.jsonl') as f:
        docs = [json.loads(x) for x in f.readlines()]
    return docs

def load_gh_so(api, lib, src):
    with open(f'data/api_db/sorted_{lib}_over10_new.jsonl') as f:
        doc_infos = [json.loads(x) for x in f.readlines()]
        
    for info in doc_infos:
        if info['api_name'] == api:
            indexes = info[src]

    docs = []
    if src == 'sos' or src == 'repos':
        key = 'id'
    else:
        key = 'number'
    with open(f'data/api_db/{src}_{lib}.jsonl') as f:
        r_docs = [json.loads(x) for x in f.readlines()]
    for doc in r_docs:
        for idx in indexes:
            if doc[key] == idx:
                if src == 'sos':
                    new_doc = {'title': doc['title'],
                            'document': doc['question'] + doc['answer']}
                elif src == 'repos':
                    new_doc = {'title': doc['name'],
                            'document': doc['content']}
                else:
                    new_doc = {
                        'title': doc['title'],
                        'document': doc['description'] + ' '.join(doc['comments'])
                    }
                new_doc['document'] = ' '.join(new_doc['document'].split()[:5000]).strip()
                docs.append(new_doc)
                
    return docs

def make_basic_rag_db(src):
    print('Making Basic RAG Document for', src)
    # load document
    rag_doc = load_basic_doc(src)
    
    embed_fn = MyEmbeddingFunction()
    client = chromadb.PersistentClient(path='./docs_db')
    collection = client.get_or_create_collection(
        name=src,
    )
    
    # Generate embeddings, and index titles in batches
    batch_size = 50

    # loop through batches and generated + store embeddings
    for i in tqdm(range(0, len(rag_doc), batch_size), ncols=75):

        # i_end = min(i + batch_size, len(rag_doc))
        batch = rag_doc[i : i + batch_size]

        # Replace title with "No Title" if empty string
        batch_docs = [str(doc["document"]) if str(doc["document"]) != "" else "No document" for doc in batch]
        batch_ids = [str(j+i) for j in range(len(batch))]
        batch_metadata = [dict(title=doc["title"]) for doc in batch]

        # generate embeddings
        batch_embeddings = embedding_model.encode(batch_docs)

        # upsert to chromadb
        collection.upsert(
            ids=batch_ids,
            metadatas=batch_metadata,
            documents=batch_docs,
            embeddings=batch_embeddings.tolist(),
        )
        
    collection = client.get_or_create_collection(
        name=src,
        embedding_function=embed_fn
    )


def make_api_rag_doc(lib, src):
    src = src.split('_')[-1]
    print('Making API level RAG database on', lib, src)
    with open(f'data/api_db/api_class_over_10_{lib}.jsonl') as f:
        api_list = [json.loads(x)['api'] for x in f.readlines()]
    for api in tqdm(api_list, total=len(api_list), ncols=75):
        print(api)
        rag_doc = load_gh_so(api, lib, src)
        embed_fn = MyEmbeddingFunction()
        client = chromadb.PersistentClient(path='./api_docs_db')
        api_name = api.split('.')[-1]
        if api_name.startswith('__'):
            api_name = api_name[2:]
        collection = client.get_or_create_collection(
            name=f'{api_name}_{lib}_{src}',
        )
        
        # Generate embeddings, and index titles in batches
        batch_size = 1

        # loop through batches and generated + store embeddings
        for i in tqdm(range(0, len(rag_doc), batch_size), ncols=75):

            # i_end = min(i + batch_size, len(rag_doc))
            batch = rag_doc[i : i + batch_size]

            # Replace title with "No Title" if empty string
            batch_docs = [str(doc["document"]) if str(doc["document"]) != "" else "No document" for doc in batch]
            batch_ids = [str(j+i) for j in range(len(batch))]
            batch_metadata = [dict(title=doc["title"]) for doc in batch]

            # generate embeddings
            batch_embeddings = embedding_model.encode(batch_docs)

            # upsert to chromadb
            collection.upsert(
                ids=batch_ids,
                metadatas=batch_metadata,
                documents=batch_docs,
                embeddings=batch_embeddings.tolist(),
            )


if len(sys.argv) > 1:
    lib = sys.argv[1]
if len(sys.argv) > 2:
    src = sys.argv[2]

if 'basic_rag' in src:
    make_basic_rag_db(src)
elif 'api_rag' in src:
    make_api_rag_doc(lib, src)
