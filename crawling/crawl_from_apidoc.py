import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys

def remove_non_ascii(string):
    return ''.join(char for char in string if ord(char) < 128)

def crawl_xgb():
    package = ''
    with open('data/xgb_api_list.txt', 'w') as f:
        with open('data/crawled/apidoc_xgb.jsonl', 'w') as f_dict:
            url = 'https://xgboost.readthedocs.io/en/latest/python/python_api.html'
            response = requests.get(url)
            html = BeautifulSoup(response.text, 'html.parser')
            functions = html.select('dl')
            for func in functions:
                try:
                    sig = func.find_all("dt", class_="sig sig-object py")[0].get_text().replace('', '').replace('class ', '').strip()
                    if sig.startswith('property '):
                        api = sig.replace('property ', '').split(':')[0].strip()
                    else:
                        api = sig.split('(')[0].strip()
                    if api.split('.')[-1][0].isupper():
                        package = api
                    if not api.startswith('xgboost.'):
                        api = package + '.' + api
                    nls = func.select('dd')
                    nl_descs = []
                    for nl in nls:
                        n = nl.select('p')
                        for l in n:
                            s = l.get_text()
                            nl_descs.append(s)
                    codes = func.select('pre')
                    ex_codes = []
                    for code in codes:
                        ex_codes.append(code.get_text())
                    api_doc = {
                        'title': api.strip(),
                        'signature': sig.strip(),
                        'nl_descs': nl_descs,
                        'ex_codes': ex_codes
                    }
                    f_dict.write(json.dumps(api_doc) + '\n')
                    f.write(api.strip() + '\n')
                except IndexError:
                    continue
            
def crawl_jax():
    with open('data/jax_apidoc_link.txt') as f:
        api_link_list = f.readlines()
    with open('data/jax_api_list.txt') as f:
        api_list = f.readlines()
    w_file = open('data/crawled/apidoc_jax.jsonl', 'w')
    new_link = open('jax_apidoc_link.txt', 'w')
    new_apis = open('jax_api_list.txt', 'w')
    for api, link in zip(api_list, api_link_list):
        api = api.strip()
        link = link.strip()
        response = requests.get(link)
        if response.status_code == 404:
            continue
        html = BeautifulSoup(response.text, 'html.parser')
        sig = html.select('dt')[0].get_text()
        
        nls = html.select('dd')[0].select('p')
        nl_descs = ''
        for nl in nls:
            if nl.get_text() != 'Examples':
                nl_descs += nl.get_text() + ' '
                
        try:
            code_ex = html.select('.highlight')[0].get_text()
        except IndexError:
            code_ex = ''
            
        api_doc = {
            'title': api.strip(),
            'signature': sig.strip(),
            'nl_descs': nl_descs.strip(),
            'ex_codes': code_ex.strip()
        }
        w_file.write(json.dumps(api_doc) + '\n')
        new_apis.write(api + '\n')
        new_link.write(link + '\n')
        
    w_file.close()
    new_link.close()
    new_apis.close()

def crawl_jax_api_list():
    with open('data/jax_api_list.txt', 'w') as f:
        with open('data/jax_apidoc_link.txt', 'w') as f_link:
            url = 'https://google-jax.qubitpi.org/en/latest/jax.html'
            response = requests.get(url)
            html = BeautifulSoup(response.text, 'html.parser')
            submodules = html.select('.toctree-l1')
            for sub in submodules:
                if 'jax.' in sub.get_text() and sub.get_text().endswith('module'):
                    sub_url = 'https://google-jax.qubitpi.org/en/latest/'+sub.find('a')['href']
                    sub_name = sub.find('a').get_text().split()[0]
                    response = requests.get(sub_url)
                    sub_html = BeautifulSoup(response.text, 'html.parser')
                    try:
                        tables = sub_html.select('table.autosummary.longtable.table')[0].select('tr')
                        for row in tables:
                            api = row.find('span').get_text()
                            api = f'{sub_name}.{api}'
                            l = row.find('a')['href']
                            api_link = f'https://google-jax.qubitpi.org/en/latest/{l}'
                            f.write(api + '\n')
                            f_link.write(api_link + '\n')
                    except IndexError:
                        classes = sub_html.select('dl')
                        for cl in classes:
                            class_name = cl.find('dt').get_text().replace('#', '').replace('class', '').strip()
                            apis = cl.find('dd')
                            for l in apis.get_text().splitlines():
                                if '[source]#' in l:
                                    api = l.split('(')[0]
                                    api_name = f'{class_name}.{api}'
                                    f.write(api_name + '\n')
                                    f_link.write(sub_url + '\n')
            apis = html.select('tr')
            for api in apis:
                api_name = api.select('span')[0].get_text()
                api_name = f'jax.{api_name}'
                api_link = api.find('a')['href']
                api_link = 'https://google-jax.qubitpi.org/en/latest/' + api_link
                f.write(api_name + '\n')
                f_link.write(api_link + '\n')


def crawl_sklearn_apilist():
    with open('data/sklearn_api_list.txt', 'w') as f:
        with open('data/sklearn_apidoc_link.txt', 'w') as f_link:
            url = 'https://scikit-learn.org/stable/api/index.html'
            response = requests.get(url)
            html = BeautifulSoup(response.text, 'html.parser')
            contents = html.select('tbody')[0].select('tr')
            # print(contents)
            # return
            for content in contents:
                api_name = content.select('a')[0].get_text()
                package = content.select('a')[1].get_text()
                api = f'{package}.{api_name}'
                link = content.select('a')[0].get_attribute_list('href')
                link = 'https://scikit-learn.org/stable/'+link[0].replace('../', '')
                f_link.write(link + '\n')
                f.write(api + '\n')
                
                
def crawl_sklearn():
    w_file = open('data/crawled/apidoc_sklearn.jsonl', 'w')
    with open('data/sklearn_apidoc_link.txt') as f:
        api_link_list = f.readlines()
    with open('data/sklearn_api_list.txt') as f:
        api_list = f.readlines()
    
    for link, api in zip(api_link_list, api_list):
        print(link)        

        response = requests.get(link)
        if response.status_code == 404:
            continue
        html = BeautifulSoup(response.text, 'html.parser')
        content = html.select('.bd-content')[0]
        try:
            sig_prename = content.select('.sig')[0].select('.sig-prename')[0].get_text()
            sig_name = content.select('.sig')[0].select('.sig-name')[0].get_text()
            sig_params = content.select('.sig')[0].select('.sig-param')
            params = '('
            for param in sig_params:
                par = param.get_text()
                params += f'{par}, '
            sig = sig_prename+sig_name+params
            sig = sig[:-2]+')'
        except IndexError:
            sig = ''
        try:
            nl_descs = content.select('.field-list')[0].get_text()
        except IndexError:
            try:
                nls = content.select('dd')[0].select('p')
                nl_descs = ''
                for nl in nls:
                    if nl.get_text() != 'Examples':
                        nl_descs += nl.get_text() + ' '
            except IndexError:
                nls = content.select('p')
                nl_descs = ''
                for nl in nls:
                    if nl.get_text() == 'previous':
                        break
                    nl_descs = nl.get_text() + ' '
        try:
            code_ex = content.select('.highlight')[0].get_text()
        except IndexError:
            code_ex = ''
        
        api_doc = {
            'title': api.strip(),
            'signature': sig.strip(),
            'nl_descs': nl_descs.strip(),
            'ex_codes': code_ex.strip()
        }
        w_file.write(json.dumps(api_doc) + '\n')
    w_file.close()
    
    
def crawl_tf():
    w_file = open('apidoc_tf.jsonl', 'w')
    with open('data/tf_api_list.txt') as f:
        api_list = f.readlines()
    url = 'https://www.tensorflow.org/api_docs/python/'
    
    for api in tqdm(api_list, total=len(api_list)):
        api = api.replace('.', '/').strip()
        tmp_url = url + api
        # print(tmp_url)

        response = requests.get(tmp_url)
        if response.status_code == 404:
            continue
        html = BeautifulSoup(response.text, 'html.parser')
        contents = html.select('devsite-content')
        
        for content in contents:
            title = content.select('.devsite-page-title')[0].get_text()
            bodies = content.select('.devsite-article-body')
            for body in bodies:
                signature = body.select('pre.lang-py.tfo-signature-link')[0].get_text()
                codes = body.select('pre.lang-python')
                nls = body.find_all('p')
                ex_codes = []
                nl_descs = []
                for nl in nls:
                    nl_descs.append(nl.get_text().strip())
                for code in codes:
                    ex_codes.append(code.get_text().strip())
            
        
        api_doc = {
            'title': title,
            'signature': signature,
            'nl_descs': nl_descs,
            'ex_codes': ex_codes
        }
        w_file.write(json.dumps(api_doc) + '\n')
    w_file.close()


def crawl_torch():
    w_file = open('apidoc_torch.jsonl', 'w')
    with open('data/torch_api_list.txt') as f:
        api_list = f.readlines()
    
    for api in tqdm(api_list, total=len(api_list)):
        api = api.strip()
        url = f'https://pytorch.org/docs/stable/generated/{api}.html'
        response = requests.get(url)
        if response.status_code == 404:
            continue
        html = BeautifulSoup(response.text, 'html.parser')
        title = remove_non_ascii(html.select('h1')[0].get_text())
        signature = remove_non_ascii(html.select('dt.sig.sig-object.py')[0].get_text()).strip()
        
        contents = html.select('dd')
        nl_descs = []
        # for content in contents:
        nls = contents[0].find_all('p')
        for nl in nls:
            nl_descs.append(nl.get_text())
        if contents[0].select('.highlight-default.notranslate'):
            codes = contents[0].select('.highlight-default.notranslate')[0].get_text()
        else:
            codes = ''
        api_doc = {
            'title': title,
            'signature': signature,
            'nl_descs': nl_descs,
            'ex_codes': [codes]
        }
        
        w_file.write(json.dumps(api_doc) + '\n')
    w_file.close()
        

if sys.argv[1] == 'torch':
    crawl_torch()
elif sys.argv[1] == 'tf':
    crawl_tf()
elif sys.argv[1] == 'sklearn':
    crawl_sklearn()
elif sys.argv[1] == 'jax':
    crawl_jax_api_list()
    crawl_jax()
elif sys.argv[1] == 'xgb':
    crawl_xgb()
else:
    print('wrong argv')