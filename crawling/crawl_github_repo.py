import requests
import time
import requests
import json
from dotenv import load_dotenv
load_dotenv()
import os
git_token = os.getenv('GIT_TOKEN')

def search_github_repos(query, language="Jupyter Notebook", min_stars=5, sort="updated", order="desc", max_pages=40):
    """Search GitHub repositories by query and paginate through results, filtering by minimum stars."""
    url = "https://api.github.com/search/repositories"
    all_repos = []
    headers = {
        'Authorization': f'token {git_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    for page in range(1, max_pages):
        time.sleep(1)  # Delay to avoid hitting API rate limit
        # Include stars qualifier in the query
        params = {
            'q': f'{query} language:"{language}" stars:>={min_stars}',
            'sort': sort,
            'order': order,
            'per_page': 10,
            'page': page
        }
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()  # Raises stored HTTPError, if one occurred
            data = response.json()
            all_repos.extend(data['items'])  # Add fetched repos to the list
            if 'next' not in response.links:
                break  # Stop if there are no more pages to fetch
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")  # Print HTTP error message
            break
        except requests.exceptions.RequestException as e:
            print(f"Other error occurred: {e}")  # Print other types of exceptions
            break
    print(all_repos)
    return all_repos

def main():
    deep_learning_tags = [
        "tensorflow",
        "tflite",
        "tf",
        "tensorflow-datasets",
        "tensorflow-models",
        "tfx",
        "tfjs",
        "tensorboard",
        "tf-nightly",
        "transformers",
        "scikit-learn",
        "sklearn",
        "mlxtend",
        "imbalanced-learn",
        "jax",
        "JAX",
        "flax",
        "jaxlib",
        "brax",
        "optax",
        "haiku",
        "trax",
        "objax",
        "chex",
        "xgboost",
        "xgb",
        "gradient-boosting",
        "lightgbm",
        "catboost",
        "torch",
        "keras",
        "pytorch",
        "pytorch-lightning",
        "torchvision",
        "torchaudio",
        
    ]
    token = ''  # Replace with your actual token
    count = 0
    for tag in deep_learning_tags:
        repos = search_github_repos(tag, max_pages=60, language="python", min_stars=400)
        time.sleep(20)  # Set the number of pages to fetch

        for repo in repos:
            data = {
                "repo_name": repo['name'],
                'description': repo['description'],
                'url': repo['html_url'],
                'updated_at': repo['updated_at'],
                'stars': repo['stargazers_count']
            }
            with open('../data/crawled/git_repos.jsonl', 'a') as f:
                f.write(json.dumps(data) + '\n')
        print(count)

def rm_dup_make_txt():
    with open('../data/crawled/git_repos.jsonl') as f:
        repos = [json.loads(x) for x in f.readlines()]
    
    key = []
    for d in repos:
        if d['url'] not in key:
            key.append(d['url'])
    
    with open('../data/crawled/git_repos.txt', 'w') as f:
        for k in key:
            f.write(k + '\n')
    print('total:', len(repos))
    print('survived:', len(key))
    print('ratio', len(key)/len(repos))

if __name__ == '__main__':
    main()
    # rm_dup_make_txt()