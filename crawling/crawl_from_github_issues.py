import requests
import collections
import sys
import json
from dotenv import load_dotenv
load_dotenv()
import os

class GithubAPI:
    issue_numbers = []
    git_email = os.getenv('GIT_EMAIL')
    git_token = os.getenv('GIT_TOKEN')
    auth = (git_email, git_token)
    
    raw = []

    def get_comments(self, url, lib):
        with open(f'data/crawled/issues_{lib}.jsonl') as f:
            self.issue_numbers = [json.loads(x)['number'] for x in f.readlines()]
        w_file = open(f'data/crawled/comments_{lib}.jsonl', 'a')
        # print(self.issue_numbers)
        for issue_num in self.issue_numbers:
            if issue_num >= 12586:
                continue
            params = {
                "per_page": 100,
                "page": 1,
            }
            raw = []
            tmp_url = url + f'/{issue_num}/comments'
            r = requests.get(tmp_url, params=params, auth=self.auth).json()
            # print(r)
            while (True):
                raw += r

                if len(r) == 100:
                    params["page"] += 1
                    r = requests.get(url, params=params, auth=self.auth).json()
                else:
                    break
            for e in raw:
                if isinstance(e, dict):
                    print("Checking comment " + str(e["id"]))
                    comments = collections.OrderedDict()
                    comments["id"] = e["id"]
                    comments["issue_num"] = issue_num
                    comments["body"] = e["body"]
                    
                    w_file.write(json.dumps(comments) + '\n')


    def get_issues(self, url, lib):
        w_file = open(f'data/crawled/issues_{lib}.jsonl', 'a')

        params = {
            "per_page": 100,
            "page": 1,
            "state": "closed",
        }
        r = requests.get(url, params=params, auth=self.auth).json()
        # print(r)

        while (True):
            self.raw += r

            if len(r) == 100:
                params["page"] += 1
                r = requests.get(url, params=params, auth=self.auth).json()
            else:
                break
        for e in self.raw:
            
            if isinstance(e, dict):
                print("Checking issue " + str(e["number"]))
                issue = collections.OrderedDict()
                issue["id"] = e["id"]
                issue["number"] = e["number"]
                issue["repo_url"] = e["repository_url"]
                issue["issue_url"] = e["url"]
                issue["events_url"] = e["events_url"]
                issue["state"] = e["state"]
                issue["html_url"] = e["html_url"]
                issue["title"] = e["title"]
                issue["description"] = e["body"]
                issue["comments"] = e["comments"]
                issue["created_at"] = e["created_at"]
                issue["updated_at"] = e["updated_at"]
                issue["closed_at"] = e["closed_at"]
                if not e["milestone"]:
                    issue["milestone"] = "null"
                else:
                    issue["milestone"] = e["milestone"]["title"]

                labels = []
                for label in e["labels"]:
                    label_issue = collections.OrderedDict()
                    label_issue["issue_repo_url"] = e["repository_url"]
                    label_issue["issue_id"] = e["id"]
                    label_issue["issue_number"] = e["number"]
                    label_issue["label_id"] = label["id"]
                    label_issue["label"] = label["name"]
                    labels.append(label_issue)

                issue["labels"] = labels

                w_file.write(json.dumps(issue) + '\n')



if __name__ == "__main__":
    if sys.argv[1] == 'tf':
        repo = 'https://github.com/tensorflow/tensorflow/issues'
    elif sys.argv[1] == 'torch':
        repo = 'https://github.com/pytorch/pytorch/issues'
    elif sys.argv[1] == 'sklearn':
        repo = 'https://github.com/scikit-learn/scikit-learn/issues'
    elif sys.argv[1] == 'jax':
        repo = 'https://github.com/google/jax/issues'
    elif sys.argv[1] == 'xgb':
        repo = 'https://github.com/dmlc/xgboost/issues'   
    else:
        print('wrong argv')
        exit()  

    app_source = repo.replace("//github.com", "//api.github.com/repos")

    print("="*80)
    print(sys.argv[1], sys.argv[2])
    print("="*80)

    api = GithubAPI()

    if sys.argv[2] == 'comments':
        print('Getting comments...')
        api.get_comments(app_source, sys.argv[1])
    elif sys.argv[2] == 'issues':
        print("Getting issues...")
        api.get_issues(app_source, sys.argv[1])
        
