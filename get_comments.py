import re, requests
import json
import pandas as pd
# import pymysql
import sklearn

from requests.exceptions import RequestException
# from multiprocessing import Pool

def parse_url(url, data):
    try:
        response = requests.get(url, params=data).text
        comments_json = re.search(r'(?<=fetchJSON_comment98vv119944\().*(?=\);)', response).group(0)
    except RequestException as e:
        return None
    return comments_json

def create_dataframe(url, data):
    comments_json = parse_url(url, data)
    j = json.loads(comments_json)
    comments = j['comments']
    df_comments = pd.DataFrame(comments[0])
    return df_comments

def append_dataframe(url, data, page, df_comment):
    data['page'] = page
    comments_json = parse_url(url, data)
    j = json.loads(comments_json)
    comments = j['comments']
    for comment in comments:
        df_comment = df_comment.append(pd.Series(comment), ignore_index = True)
    return df_comment

def get_comments(url, data, pages=50):
    df_comment = create_dataframe(url, data)
    for i in range(pages):
        df_comment = append_dataframe(url, data, i, df_comment)
    df_content = df_comment['content']
    return df_content

def mycut(s):
    return ' '.join(jieba.cut(s))

def main():
    url = 'https://club.jd.com/comment/productPageComments.action'
    data = {
        'callback': 'fetchJSON_comment98vv119944',
        'productId': '4586850',
        'score': 1,
        'sortType': 6,
        'pageSize': 10,
        'isShadowSku': 0,
        'page': 0,
        'fold': 1
    }
    df_content = get_comments(url, data, 100)
