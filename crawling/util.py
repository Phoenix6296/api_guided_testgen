from html2text import html2text
import json
from tqdm import tqdm
import sys
import statistics
import re
import os

# remove duplicated instances from SO doc
def rm_dup_so(lib='tf'):
    with open(f'data/crawled/so_{lib}.jsonl') as f:
        records = [json.loads(l) for l in f.readlines()]
    
    keys = []
    uniques = []
    for rec in tqdm(records, total=len(records)):
        if rec['link'] not in keys:
            keys.append(rec['link'])
            uniques.append(rec)
            
    with open(f'so_{lib}_rmdup.jsonl', 'w') as f:
        for lin in tqdm(uniques, total=len(uniques)):
            f.write(json.dumps(lin) + '\n')


# filter out irrelevant comments from issue doc
def filter_comments(lib='tf'):
    with open(f'data/crawled/comments_{lib}.jsonl') as f:
        records = [json.loads(l) for l in f.readlines()]
    
    w_file = open(f'data/crawled/filtered_comments_{lib}.jsonl', 'w')
    
    if lib=='tf':
        for e in records:
            if not e['body']:
                continue
            if  e['body'] and \
                e['body'].startswith("Thanks for your pull request! It looks like this may be your first contribution to a Google") or \
                e['body'].startswith("Thanks everyone, it's been solved.") or \
                e['body'].startswith("Are you satisfied with the resolution of your issue?") or \
                e['body'].startswith("Thank you for your response.") or \
                e['body'].startswith("This issue is stale because it has been open for 7 days with no activity.") or \
                e['body'].startswith("This issue was closed because it has been inactive for 7 days") or \
                e['body'].startswith('Automated Code Change\n'):
                continue
            w_file.write(json.dumps(e) + '\n')
    elif lib=='torch':
        for e in records:
            if not e['body']:
                continue
            if e['body'] and \
                e['body'].startswith("<!-- drci-comment-start -->\n\n## :link: Helpful Links") or \
                e['body'].startswith(" <a href=\"https://api.easycla.lfx.linuxfoundation.org") or \
                e['body'].startswith("@pytorchbot") or \
                e['body'].startswith("### Merge") or \
                e['body'].startswith("## Merge") or \
                e['body'].startswith('Closing in favor of') or \
                e['body'].startswith('This pull request was exported from') or \
                e['body'].startswith('The merge [job](https://github.com/pytorch/') or \
                e['body'].startswith('@pytorchmergebot') or \
                e['body'].startswith('This pull request was **exported** from Phabricator.') or \
                e['body'].startswith('Successfully rebased') or \
                e['body'].startswith('<a href=\"https://easycla.lfx.linuxfoundation.org') or \
                e['body'].startswith('Rebase failed due to Command') or \
                e['body'].startswith('/assigntome') or \
                e['body'].startswith('Please seek CI approval before scheduling'):
                continue
            w_file.write(json.dumps(e) + '\n')
    elif lib=='sklearn':
        for e in records:
            if not e['body']:
                continue
            if 'Linting Passed\nAll linting checks passed.' in e['body']:
                continue
            w_file.write(json.dumps(e) + '\n')
    elif lib=='jax':
        for e in records:
            if not e['body']:
                continue
            if e['body'] and \
                e['body'].startswith("Thanks for the PR!") or \
                e['body'].startswith("I signed it!") or \
                e['body'].startswith('Closing as stale.') or \
                e['body'].startswith('Closing as a duplicate of'):
                continue
            w_file.write(json.dumps(e) + '\n')
    elif lib == 'xgb':
        for e in records:
            if not e['body']:
                continue
            if e['body'] and \
                e['body'].startswith("cc @rongou .") or \
                e['body'].startswith("OK, I won't notify you again about this release, but will get in touch") or \
                e['body'].startswith('@dependabot rebase') or \
                e['body'].startswith('Closing as a duplicate of'):
                continue
            w_file.write(json.dumps(e) + '\n')
    w_file.close()


# merge mined issue and its comments
def merge_comments_n_issues(lib='tf'):
    with open(f'data/crawled/issues_{lib}.jsonl') as f:
        issues = [json.loads(l) for l in f.readlines()]
    with open(f'data/crawled/filtered_comments_{lib}.jsonl') as f:
        comments = [json.loads(l) for l in f.readlines()]
    
    w_file = open(f'data/crawled/issues_n_comments_{lib}.jsonl', 'w')

    for issue in tqdm(issues, total=len(issues)):
        for comment in comments:
            if comment['issue_num'] == issue['number']:
                if isinstance(issue['comments'], int):
                    issue['comments'] = [comment['body']]
                elif isinstance(issue['comments'], list):
                    issue['comments'].append(comment['body'])
    
        w_file.write(json.dumps(issue) + '\n')
    
    w_file.close()


# get the sorted numbers of apis in a doc from src
def count_most_used(lib, src):
    with open(f'data/api_db/{src}_{lib}.jsonl') as f:
        docs = [json.loads(l) for l in f.readlines()]
    
    api_cnts = {}
    for doc in docs:
        for api in doc['apis']:
            if api in api_cnts:
                api_cnts[api] += 1
            else:
                api_cnts[api] = 1
    # sort by counts most api counts
    api_cnts = dict(sorted(api_cnts.items(), key=lambda item: item[1], reverse=True))
    api_cnts['total_cnts'] = len(api_cnts)
    with open(f'data/stats/{lib}_{src}.json', 'w') as f:
        f.write(json.dumps(api_cnts))


# get the sorted number of apis considering both docs
def count_most_both(lib):
    with open(f'data/stats/{lib}_so.json') as f:
        sos = json.loads(f.read())
    with open(f'data/stats/{lib}_issues_n_comments.json') as f:
        issues = json.loads(f.read())
    with open(f'data/stats/{lib}_repos.json') as f:
        repos = json.loads(f.read())
    combined = {}
    for so_k, so_v in sos.items():
        for iss_k, iss_v in issues.items():
            for rep_k, rep_v in repos.items():
                if so_k == iss_k == rep_k:
                    combined[so_k] = so_v + iss_v + rep_v
    
    combined['total_cnts'] = len(combined)
    with open(f'data/stats/{lib}_combined_new.json', 'w') as f:
        f.write(json.dumps(combined))


def get_api_related_repo():
    src = 'repos'
    for lib in ['tf', 'torch', 'sklearn', 'xgb', 'jax']:
        with open(f'data/api_db/{src}_{lib}.jsonl') as f:
            repos = [json.loads(x) for x in f.readlines()]
        with open(f'data/crawled/apidoc_{lib}.jsonl') as f:
            api_list = [json.loads(x)['title']+'(' for x in f.readlines()]
        api_repos = []
        for repo in tqdm(repos, total=len(repos), ncols=70):
            for api in api_list:
                search_api = api.split('.')[-1]
                if search_api in repo['document']:
                    if api_repos and any(d['id'] == repo['id'] for d in api_repos):
                        for d in api_repos:
                            if d['id'] == repo['id']:
                                d['apis'].append(api)
                    else:
                        san_repo = {
                            'id': repo['id'],
                            'name': repo['title'],
                            'content': repo['document'],
                            'apis': [api]
                        }
                        api_repos.append(san_repo)
        with open(f'data/api_db/{src}_{lib}_.jsonl', 'w') as f:
            for r in tqdm(api_repos, total=len(api_repos), ncols=70):
                f.write(json.dumps(r) + '\n')


# filter SO doc with api related
def get_api_related_so(lib='tf'):
    src='sos'
    with open(f'data/crawled/{src}_{lib}.jsonl', 'r') as f:
        so_questions = [json.loads(x) for x in f.readlines()]
    with open(f'data/crawled/apidoc_{lib}.jsonl') as f:
        api_list = [json.loads(x)['title']+'(' for x in f.readlines()]
    
    api_sos = []
    for so in tqdm(so_questions, total=len(so_questions), ncols=70):
        for api in api_list:
            search_api = api.split('.')[-1]
            question = html2text(so['question'])
            answer = html2text(so['answer'])
            # if hit (has api in one of the infos)
            if search_api in so['title'] or \
                search_api in question or \
                search_api in answer:
                # if already added by checking the id
                if api_sos and any(d['id'] == so['id'] for d in api_sos):
                    for d in api_sos:
                        if d['id'] == so['id']:
                            d['apis'].append(api)
                # if new
                else:
                    san_so = {
                        'id': so['id'],
                        'title': so['title'],
                        'question': question,
                        'answer': answer,
                        'apis': [api]
                    }
                    api_sos.append(san_so)
            
    with open(f'data/api_db/{src}_{lib}.jsonl', 'w') as f:
        for q in tqdm(api_sos, total=len(api_sos), ncols=70):
            f.write(json.dumps(q) + '\n')


# filter github issue doc with api related
def get_api_related_issues(lib='tf'):
    src = 'issues_n_comments'
    with open(f'data/crawled/{src}_{lib}.jsonl', 'r') as f:
        issues = [json.loads(x) for x in f.readlines()]
    with open(f'data/crawled/apidoc_{lib}.jsonl') as f:
        api_list = [json.loads(x)['title']+'(' for x in f.readlines()]
    
    api_issues = []
    for issue in tqdm(issues, total=len(issues), ncols=70):
        if issue['description'] and isinstance(issue['comments'], list):
            comments = '\n===\n'.join(issue['comments'])
            for api in api_list:
                search_api = api.split('.')[-1]
                # if hit (has api in one of the infos)
                if search_api in issue['title'] or \
                    search_api in issue['description'] or \
                    search_api in comments:
                    # if issue is already added by checking the issue number:
                    if api_issues and any(d['number'] == issue['number'] for d in api_issues):
                        for d in api_issues:
                            if d['number'] == issue['number']:
                                d['apis'].append(api)
                    # if new
                    else:
                        san_issue = {
                            'number': issue['number'],
                            'title': issue['title'],
                            'description': issue['description'],
                            'comments': issue['comments'],
                            'apis': [api]
                        }
                        api_issues.append(san_issue)
    with open(f'data/api_db/{src}_{lib}.jsonl', 'w') as f:
        for q in tqdm(api_issues, total=len(api_issues), ncols=70):
            f.write(json.dumps(q) + '\n')


# combine API_DB in api level
def combine_api_db(lib='tf'):
    with open(f'data/stats/{lib}_combined_new.json') as f:
        api_dicts = json.loads(f.read())
    apis = list(api_dicts.keys())[:-1]
    with open(f'data/api_db/issues_{lib}.jsonl') as f:
        issues = [json.loads(l) for l in f.readlines()]
    with open(f'data/api_db/sos_{lib}.jsonl') as f:
        sos = [json.loads(l) for l in f.readlines()]
    with open(f'data/api_db/repos_{lib}.jsonl') as f:
        repos = [json.loads(l) for l in f.readlines()]
        
    w_file = open(f'data/api_db/combined_{lib}_new.jsonl', 'w')
    api_docs = []
    for api in tqdm(apis, total=len(apis), ncols=70):
        api_doc = {}
        api_doc['api_name'] = api[:-1]
        api_doc['sos'] = []
        api_doc['issues'] = []
        api_doc['repos'] = []
        for iss in issues:
            if api in iss['apis']:
                api_doc['issues'].append(iss['number'])
        api_doc['issue_cnt'] = len(api_doc['issues'])
        for so in sos:
            if lib == 'sklearn':
                api = api.split('.')[-1]
            if api in so['apis']:
                api_doc['sos'].append(so['id'])
        api_doc['so_cnt'] = len(api_doc['sos'])
        for repo in repos:
            if api in repo['apis']:
                api_doc['repos'].append(repo['id'])
        api_doc['repo_cnt'] = len(api_doc['repos'])
        # api_doc['total_cnt'] = api_dicts[api]
        api_docs.append(api_doc)
    for w in api_docs:
        w_file.write(json.dumps(w) + '\n')
    w_file.close()


def get_combined_apidoc(lib='tf'):
    with open(f'data/stats/{lib}_combined_new.json') as f:
        api_dicts = json.loads(f.read())
    apis = list(api_dicts.keys())[:-1]

    with open(f'data/crawled/apidoc_{lib}.jsonl') as f:
        api_docs = [json.loads(l) for l in f.readlines()]

    with open(f'data/api_db/apidoc_{lib}.jsonl', 'w') as f:
        uniq = []
        for api in apis:
            for api_doc in api_docs:
                if api[:-1] in api_doc['title'] and api_doc['title'] not in uniq:
                    uniq.append(api_doc['title'])
                    f.write(json.dumps(api_doc) + '\n')


def generate_basic_rag_jsonl():
    libs = ['tf','torch','sklearn', 'jax', 'xgb']
    srcs = ['apidoc', 'issues', 'sos', 'repos']
    all_data = []
    for lib in libs:
        for src in srcs:
            with open(f'data/api_db/{src}_{lib}.jsonl') as f:
                db = [json.loads(l) for l in f.readlines()]
                dbd = {
                    'lib': lib,
                    'src': src,
                    'docs': db
                }
                all_data.append(dbd)
    api_f = open('data/api_db/basic_rag_apidoc.jsonl', 'w')
    issues_f = open('data/api_db/basic_rag_issues.jsonl', 'w')
    sos_f = open('data/api_db/basic_rag_sos.jsonl', 'w')
    repos_f = open('data/api_db/basic_rag_repos.jsonl', 'w')
    all_f = open('data/api_db/basic_rag_all.jsonl', 'w')
    for data in tqdm(all_data, total=len(all_data), ncols=70):
        if data['src'] == 'apidoc':
            for d in data['docs']:
                api_name = d['title']
                if isinstance(d['nl_descs'], list):
                    nl = ' '.join(d['nl_descs'])
                else:
                    nl = d['nl_descs']
                if isinstance(d['ex_codes'], list):
                    code = ' '.join(d['ex_codes'])
                else:
                    code = d['ex_codes']
                doc = {
                    'title': api_name,
                    'document': f"Signature: {d['signature']} \n Description: {nl} \n Example: {code}"
                }
                api_f.write(json.dumps(doc) + '\n')
                all_f.write(json.dumps(doc) + '\n')
        elif data['src'] == 'issues':
            for d in data['docs']:
                doc = {
                    'title': d['title'],
                    'document': 'Description: ' + d['description'] + ' ' + '\ncomments: '.join(d['comments'])
                }
                doc['document'] = ' '.join(doc['document'].split()[:5000]).strip()
                issues_f.write(json.dumps(doc) + '\n')
                all_f.write(json.dumps(doc) + '\n')
        elif data['src'] == 'sos':
            for d in data['docs']:
                doc = {
                    'title': d['title'],
                    'document': 'Question: '+ d['question'] + ' Answer: ' + d['answer']
                }
                doc['document'] = ' '.join(doc['document'].split()[:5000]).strip()
                sos_f.write(json.dumps(doc) + '\n')
                all_f.write(json.dumps(doc) + '\n')
        elif data['src'] == 'repos':
            for d in data['docs']:
                doc = {
                    'title': d['name'],
                    'document': d['content']
                }
                doc['document'] = ' '.join(doc['document'].split()[:5000]).strip()
                repos_f.write(json.dumps(doc) + '\n')
                all_f.write(json.dumps(doc) + '\n')
    api_f.close()
    issues_f.close()
    sos_f.close()
    repos_f.close()
    all_f.close()


def generate_basic_rag_combined():
    with open('data/api_db/apidoc_tf.jsonl') as f:
        tf_apidoc = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/apidoc_torch.jsonl') as f:
        torch_apidoc = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/issues_n_comments_tf.jsonl') as f:
        tf_issues = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/issues_n_comments_torch.jsonl') as f:
        torch_issues = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/sos_tf.jsonl') as f:
        tf_sos = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/sos_torch.jsonl') as f:
        torch_sos = [json.loads(l) for l in f.readlines()]
    tf_api_names = []
    for api in tf_apidoc:
        tf_api_names.append(api['title']+'(')
    torch_api_names = []
    for api in torch_apidoc:
        torch_api_names.append(api['title']+'(')
    with open('data/api_db/basic_rag_doc.jsonl', 'w') as f:
        for api in tf_apidoc:
            api_name = api['title']
            doc = {
                'title': api_name, 
                'document': api['signature'] + ' '.join(api['nl_descs']) + ' '.join(api['ex_codes'])
            }
            f.write(json.dumps(doc) + '\n')
        for issue in tf_issues:
            if any(check in issue['apis'] for check in tf_api_names):
                doc = {
                    'title': issue['title'], 
                    'document': issue['description'] + ' '.join(issue['comments'])
                }
                f.write(json.dumps(doc) + '\n')
        for so in tf_sos:
            if any(check in so['apis'] for check in tf_api_names):
                doc = {'title': so['title'], 
                        'document': so['question'] +
                        so['answer']}
                f.write(json.dumps(doc) + '\n')
        for api in torch_apidoc:
            api_name = api['title']
            doc = {
                'title': api_name,
                'document': api['signature'] + ' '.join(api['nl_descs']) + ' '.join(api['ex_codes'])
            }
            f.write(json.dumps(doc) + '\n')
        for issue in torch_issues:
            if any(check in issue['apis'] for check in torch_api_names):
                doc = {'title': issue['title'],
                        'document': issue['description'] +
                        ' '.join(issue['comments'])}
                f.write(json.dumps(doc) + '\n')
        for so in torch_sos:
            if any(check in so['apis'] for check in torch_api_names):
                doc = {'title': so['title'],
                        'document': so['question'] +
                        so['answer']}
                f.write(json.dumps(doc) + '\n')


def split_list(lst):
    split_point = len(lst) * 9 // 10
    return lst[:split_point], lst[split_point:]


def generate_basic_rag_git_jsonl():
    libs = ['tf', 'torch', 'sklearn', 'jax', 'xgb']
    for lib in libs:
        data = []
        id = 0
        for cur, dirs, files in os.walk(f'crawling/classified_files/{lib}'):
            for file in files:
                if file.endswith('py'):
                    with open(f'{cur}/{file}') as f:
                        file_content = f.read()
                    d = {'id': id, 'title': file, 'document': file_content}
                    data.append(d)
                    id +=1
        with open(f'data/api_db/repos_{lib}.jsonl', 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')


def generate_basic_rag_combined_train_test_split():
    with open('data/api_db/apidoc_tf.jsonl') as f:
        tf_apidoc = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/apidoc_torch.jsonl') as f:
        torch_apidoc = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/issues_n_comments_tf.jsonl') as f:
        tf_issues = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/issues_n_comments_torch.jsonl') as f:
        torch_issues = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/sos_tf.jsonl') as f:
        tf_sos = [json.loads(l) for l in f.readlines()]
    with open('data/api_db/sos_torch.jsonl') as f:
        torch_sos = [json.loads(l) for l in f.readlines()]
    tf_api_names = []
    for api in tf_apidoc:
        tf_api_names.append(api['title']+'(')
    torch_api_names = []
    for api in torch_apidoc:
        torch_api_names.append(api['title']+'(')
    tf_apidoc_train, tf_apidoc_test = split_list(tf_apidoc)
    tf_issues_train, tf_issues_test = split_list(tf_issues)
    tf_sos_train, tf_sos_test = split_list(tf_sos)
    torch_apidoc_train, torch_apidoc_test = split_list(torch_apidoc)
    torch_issues_train, torch_issues_test = split_list(torch_issues)
    torch_sos_train, torch_sos_test = split_list(torch_sos)
    with open('data/api_db/basic_rag_all_train.jsonl', 'w') as f_train:
        with open('data/api_db/basic_rag_all_test.jsonl', 'w') as f_test:
            for api in tf_apidoc_train:
                api_name = api['title']
                doc = {'title': api_name, 
                    'document': api['signature'] +
                    ' '.join(api['nl_descs']) +
                    ' '.join(api['ex_codes'])}
                f_train.write(json.dumps(doc) + '\n')
            for issue in tf_issues_train:
                if any(check in issue['apis'] for check in tf_api_names):
                    doc = {'title': issue['title'], 
                            'document': issue['description'] +
                            ' '.join(issue['comments'])}
                    f_train.write(json.dumps(doc) + '\n')
            for so in tf_sos_train:
                if any(check in so['apis'] for check in tf_api_names):
                    doc = {'title': so['title'], 
                            'document': so['question'] +
                            so['answer']}
                    f_train.write(json.dumps(doc) + '\n')
                        
            for api in torch_apidoc_train:
                api_name = api['title']
                doc = {'title': api_name, 
                    'document': api['signature'] +
                    ' '.join(api['nl_descs']) +
                    ' '.join(api['ex_codes'])}
                f_train.write(json.dumps(doc) + '\n')
            for issue in torch_issues_train:
                if any(check in issue['apis'] for check in torch_api_names):
                    doc = {'title': issue['title'], 
                            'document': issue['description'] +
                            ' '.join(issue['comments'])}
                    f_train.write(json.dumps(doc) + '\n')
            for so in torch_sos_train:
                if any(check in so['apis'] for check in torch_api_names):
                    doc = {'title': so['title'], 
                            'document': so['question'] +
                            so['answer']}
                    f_train.write(json.dumps(doc) + '\n')
                    
            for api in tf_apidoc_test:
                api_name = api['title']
                doc = {'title': api_name, 
                    'document': api['signature'] +
                    ' '.join(api['nl_descs']) +
                    ' '.join(api['ex_codes'])}
                f_test.write(json.dumps(doc) + '\n')
            for issue in tf_issues_test:
                if any(check in issue['apis'] for check in tf_api_names):
                    doc = {'title': issue['title'], 
                            'document': issue['description'] +
                            ' '.join(issue['comments'])}
                    f_test.write(json.dumps(doc) + '\n')
            for so in tf_sos_test:
                if any(check in so['apis'] for check in tf_api_names):
                    doc = {'title': so['title'], 
                            'document': so['question'] +
                            so['answer']}
                    f_test.write(json.dumps(doc) + '\n')
            for api in torch_apidoc_test:
                api_name = api['title']
                doc = {'title': api_name, 
                    'document': api['signature'] +
                    ' '.join(api['nl_descs']) +
                    ' '.join(api['ex_codes'])}
                f_test.write(json.dumps(doc) + '\n')
            for issue in torch_issues_test:
                if any(check in issue['apis'] for check in torch_api_names):
                    doc = {'title': issue['title'], 
                            'document': issue['description'] +
                            ' '.join(issue['comments'])}
                    f_test.write(json.dumps(doc) + '\n')
            for so in torch_sos_test:
                if any(check in so['apis'] for check in torch_api_names):
                    doc = {'title': so['title'], 
                            'document': so['question'] +
                            so['answer']}
                    f_test.write(json.dumps(doc) + '\n')


def cnt_total_tests(lib='tf'):
    with open(f'{lib}_eval.log') as f:
        lines = f.readlines()
    sum_ = 0
    for line in lines:
        if line.startswith('test:'):
            sum_ += int(line.split('test: ')[-1].strip())
    print(sum_)


def get_with_over_10_docs(lib='tf'):
    with open(f'data/api_db/combined_{lib}_new.jsonl') as f:
        apis = [json.loads(x) for x in f.readlines()]
    selected = []
    for a in apis:
        score = [a['so_cnt'], a['issue_cnt'], a['repo_cnt']]
        harm_mean = statistics.harmonic_mean(score)
        a['score'] = harm_mean
        if a['so_cnt'] > 10 and a['issue_cnt'] > 10:
            selected.append(a)
    sorted_list = sorted(selected, key=lambda selected: selected['score'], reverse=True)
    with open(f'data/api_db/sorted_{lib}_over10_new.jsonl', 'w') as f:
        for w in sorted_list:
            f.write(json.dumps(w) + '\n')


def get_buggy_issues(lib):
    with open(f'../data/crawled/issues_{lib}.jsonl') as f:
        issues = [json.loads(x) for x in f.readlines()]
    bug_issue = []
    for i, issue in enumerate(issues):
        if issue['labels']:
            for label in issue['labels']:
                if label['label'] in ['type:bug', 'bug', 'dynamo-must-fix',
                                    'module: debug-build', 'module: error checking']:
                    bug_issue.append(issue)
                    break
    with open(f'../data/crawled/buggy_issues_{lib}.jsonl', 'w') as f:
        for l in bug_issue:
            f.write(json.dumps(l) + '\n')


def parse_issue(issue):
    # Extract basic information
    result = {
        'issue_number': issue['number'],
        'title': issue['title'],
        'version': None,
        'api': None,
        'symptom': None,
        'replication': None
    }
    description = issue.get('description', '')
    # Parse TensorFlow version
    version_match = re.search(r'### TensorFlow version\s*\n\s*([^\n]+)', description)
    if version_match:
        result['version'] = version_match.group(1).strip().replace('tf ', '').replace('v', '')
    # Parse symptom (current behavior)
    symptom_match = re.search(r'### Current behavior\?\s*\n\s*([^\n]+(?:\n\s*[^\n]+)*)', description)
    if symptom_match:
        result['symptom'] = symptom_match.group(1).strip()
    # Parse replication code
    replication_match = re.search(r'### Standalone code to reproduce the issue\s*\n\s*```.*?\n(.*?)```', description, re.DOTALL)
    if replication_match:
        result['replication'] = replication_match.group(1).strip()
    # Detect API from title and description
    api_pattern = r'\b(tf\.\w+|tensorflow\.\w+|tflite)\b'
    api_candidates = set()
    # Check title
    api_candidates.update(re.findall(api_pattern, issue['title']))
    # Check description
    api_candidates.update(re.findall(api_pattern, description))
    if api_candidates:
        # Prefer candidates mentioned in both title and error
        error_messages = re.findall(r'Error:\s*(.*?)\n', description)
        for candidate in api_candidates:
            if any(candidate in msg for msg in error_messages):
                result['api'] = candidate
                break
        if not result['api']:
            result['api'] = list(api_candidates)[0]
    return result

def make_bug_data(lib):
    issues = []
    with open(f'buggy_issues_{lib}.jsonl', 'r') as f:
        for line in f:
            issues.append(json.loads(line))
    # Parse all issues
    parsed_issues = [parse_issue(issue) for issue in issues]
    # Print results
    with open(f'buggy_data_{lib}.jsonl', 'w') as f:
        for idx, parsed in enumerate(parsed_issues):
            parsed['id'] = idx
            f.write(json.loads(parsed) + '\n')


# tf/torch so/issues_n_comments/issues
if len(sys.argv) >= 2:
    lib=sys.argv[1]
if len(sys.argv) >= 3:
    src=sys.argv[2]


# cnt_total_tests(lib)
# generate_basic_rag_jsonl()
# generate_basic_rag_git_jsonl()
# get_api_related_repo()
# rm_dup_so(lib)
# filter_comments(lib)
# merge_comments_n_issues(lib)

# # to make api basic dataset
# get_api_related_issues(lib)
# get_api_related_so(lib)

# count_most_used(lib, 'issues_n_comments')
# count_most_used(lib, 'so')
# count_most_used(lib, 'repos')
count_most_both(lib)
combine_api_db(lib)
# get_combined_apidoc(lib)

get_with_over_10_docs(lib)

# # needed to make buggy data
# make_bug_data(lib)
# get_buggy_issues(lib)