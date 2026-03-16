import requests
import time
import json
import sys
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv('STACKOVERFLOW_KEY')


def fetch_questions(tag, site='stackoverflow'):
    """Fetch questions tagged with a specific tag from Stack Overflow and handle pagination."""
    all_questions = []
    
    for page in range(8, 30):
        print("page, ", page)
        time.sleep(2)
        url = f'https://api.stackexchange.com/2.3/questions'
        params = {
            'page': page,
            'pagesize': 100,
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'site': site,
            'filter': 'withbody',
            'key': api_key
        }
        response = requests.get(url, params=params)
        # print(response)
        try:
            response.raise_for_status()  # Raises an HTTPError for bad requests
            data = response.json()
            all_questions.extend(data['items'])
            if not data.get('has_more', False):  # Check if there are more pages
                break
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")  # Print HTTP error message
            break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")  # Print any other request issues
            continue
    return all_questions


def fetch_accepted_answer(question_id, site='stackoverflow'):
    url = f'https://api.stackexchange.com/2.3/questions/{question_id}/answers'
    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': site,
        'filter': 'withbody',
        'is_accepted': True,
        'key': api_key
    }
    max_retries = 5  # Maximum number of retries
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            time.sleep(1)
            response.raise_for_status()  # Will raise an exception for HTTP errors
            answers = response.json()
            for answer in answers.get('items', []):
                if answer.get('is_accepted'):
                    print("finally")
                    return answer
            return None  # No accepted answer found
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("Max retries exceeded")
                return None


def main(lib):
    start_from = 0
    w_file = open(f'data/crawled/sos_{lib}.jsonl', 'a')
    if lib == 'sklearn':
        tags = ['scikit-learn']
    elif lib == 'torch':
        tags = ['pytorch']
    elif lib == 'tf':
        tags = ['tensorflow']
    elif lib == 'jax':
        tags = ['jax', 'google-jax']
    elif lib == 'xgb':
        tags = ['xgboost', 'xgbclassifier', 'xgbregressor']
    
    for tag in tags:
        questions = fetch_questions(tag)
        count = start_from
        idx = 0
        for question in questions:
            idx += 1
            if idx <= start_from: continue
            if question.get('is_answered') and question.get('accepted_answer_id'):
                accepted_answer = fetch_accepted_answer(question['question_id'])
                
                if accepted_answer:
                    count+= 1
                    link = question["link"]
                    answer = accepted_answer["body"]
                    title = question["title"]
                    q = question["body"]
                    data = {
                        "id": count,
                        "link": link,
                        "title": title,
                        "question": q,
                        "ansewr": answer
                    }
                    w_file.write(json.dumps(data) + '\n')
                print("-" * 80)
                print(count)
    w_file.close()

if __name__ == '__main__':
    lib = sys.argv[1]
    main(lib)

