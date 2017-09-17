import re, requests
import json
import pandas as pd
# import pymysql
from requests.exceptions import RequestException
# from multiprocessing import Pool
import jieba
from gensim import corpora, models

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

def get_comments(url, data, pages):
    df_comment = create_dataframe(url, data)
    for i in range(pages):
        df_comment = append_dataframe(url, data, i, df_comment)
    df_content = df_comment['content']
    return df_content

def mycut(s):
    return ' '.join(jieba.cut(s))

def cut_sentence(df):
    cut_1 = df[0].apply(mycut)
    stoplist = 'stoplist.txt'
    stop = pd.read_csv(stoplist, encoding = 'utf-8', header = None, sep = 'tipdm', engine='python')
    # sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停用词表中，因此会导致读取出错
    # 所以解决办法是手动设置一个不存在的分割词，如tipdm。
    stop = [' ', ''] + list(stop[0])  # Pandas自动过滤了空格符，这里手动添加
    cut_2 = cut_1.apply(lambda s: s.split(' ')) #定义一个分割函数，然后用apply广播
    cut_3 = cut_2.apply(lambda x: [i for i in x if i not in stop]) #逐词判断是否停用词，思路同上
    # df_cut = pd.DataFrame(cut_3)
    return cut_3

def print_topics(cut_3, num=50):
    dict = corpora.Dictionary(cut_3) #建立词典
    corpus = [dict.doc2bow(i) for i in cut_3] #建立语料库
    lda = models.LdaModel(corpus, num_topics = num, id2word = dict) #LDA模型训练
    for i in range(lda.num_topcis):
        print(lda.print_topic(i))

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
    cut_content = cut_sentence(df_content)
    print_topics(cut_content)

if __name__ == '__main__':
    main()