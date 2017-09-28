import re, requests
import json
import pandas as pd
from requests.exceptions import RequestException
import jieba
from gensim import corpora, models

import pymysql
from multiprocessing import Pool


def parse_url(url, data):
    '''
    fetchJSON_comment
    :param url: url
    :param data: Query String Parameters
    :return: fetchJSON_comment
    '''
    pattern = r'(?<=' + data['callback'] + '\().*(?=\);)'
    try:
        response = requests.get(url, params=data).text
        comments_json = re.search(pattern, response).group(0)
    except RequestException as e:
        return None
    return comments_json


def create_dataframe(url, data):
    '''
    parse fetchJSON_comment to create dataframe
    :param url: url
    :param data: Query String Parameters
    :return: create an empty dataframe
    '''
    comments_json = parse_url(url, data)
    j = json.loads(comments_json)
    comments = j['comments']
    df_comments = pd.DataFrame(comments[0])
    return df_comments


def append_dataframe(url, data, page, df_comment):
    '''
    append the dataframe from the selected page
    :param url: url
    :param data: Query String Parameters
    :param page: page
    :param df_comment: the original dataframe
    :return: appended dataframe
    '''
    data['page'] = page
    comments_json = parse_url(url, data)
    j = json.loads(comments_json)
    comments = j['comments']
    for comment in comments:
        df_comment = df_comment.append(pd.Series(comment), ignore_index = True)
    return df_comment


def get_comments(url, data, pages):
    '''
    get comments dataframe from input url, query string and number of pages
    :param url: url
    :param data: Query String Parameters
    :param pages: number of pages to be crawled
    :return: dataframe that contains the comments
    '''
    df_comment = create_dataframe(url, data)
    for i in range(pages):
        df_comment = append_dataframe(url, data, i, df_comment)
    df_content = df_comment['content']
    return df_content


def mycut(s):
    # cut comment sentence with jieba
    return ' '.join(jieba.cut(s))


def cut_sentence(df):
    cut_1 = df.apply(mycut) # 用jieba库做分词
    stop_list = 'stoplist.txt'
    stop = pd.read_csv(stop_list, encoding='utf-8', header=None, sep='tipdm', engine='python')
    # sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停用词表中，因此会导致读取出错
    # 所以解决办法是手动设置一个不存在的分割词，如tipdm。
    stop = [' ', ''] + list(stop[0])  # Pandas自动过滤了空格符，这里手动添加
    cut_2 = cut_1.apply(lambda s: s.split(' ')) #定义一个分割函数，然后用apply广播
    cut_3 = cut_2.apply(lambda x: [i for i in x if i not in stop]) #逐词判断是否停用词，思路同上
    # df_cut = pd.DataFrame(cut_3)
    return cut_3


def print_topics(cut_3, num=50):
    '''
    print topics of given number
    :param cut_3: material
    :param num: number of topics, default: 50
    :return: no return, just print topics
    '''
    dicts = corpora.Dictionary(cut_3)  # 建立词典
    corpus = [dicts.doc2bow(i) for i in cut_3]  # 建立语料库
    lda = models.LdaModel(corpus, num_topics=num, id2word=dicts)  # LDA模型训练
    for i in range(lda.num_topics):
        print(lda.print_topic(i))


def main():
    url = 'https://club.jd.com/comment/productPageComments.action'
    # callback = 'fetchJSON_comment98vv119944'
    # productId = '4586850'
    callback = 'fetchJSON_comment98vv2167'  # modify this string based on specified item
    productId = '4431213'  # productId
    data = {
        'callback': callback,
        'productId': productId,
        'score': 2,
        'sortType': 6,
        'pageSize': 10,
        'isShadowSku': 0,
        'page': 0,
        'fold': 1
    }
    df_content = get_comments(url, data, 20)
    cut_content = cut_sentence(df_content)
    print_topics(cut_content, 10)

if __name__ == '__main__':
    main()