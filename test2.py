from sec_api import QueryApi
from dotenv import load_dotenv
import os,requests
#import ssl
# pip install unstructured
#from unstructured.partition.html import partition_html
from bs4 import BeautifulSoup
import yfinance as yf

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()



def __download_form_html(url):
    headers = {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
      'Cache-Control': 'max-age=0',
      'Dnt': '1',
      'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
      'Sec-Ch-Ua-Mobile': '?0',
      'Sec-Ch-Ua-Platform': '"macOS"',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    return response.text

# sec_api_key = os.getenv("SEC_API_KEY")
# queryApi = QueryApi(api_key=os.environ['SEC_API_KEY'])

# print(f"step 1 : input data")
# data = "AAPL|what was last quarter's revenue"
# stock, ask = data.split("|")
# print(f"stock: {stock}, ask: {ask}")

# print(f"step 2 : query sec api and get filings and url")
# query = {
#       "query": {
#         "query_string": {
#           "query": f"ticker:{stock} AND formType:\"10-Q\""
#         }
#       },
#       "from": "0",
#       "size": "1",
#       "sort": [{ "filedAt": { "order": "desc" }}]
#     }

# fillings = queryApi.get_filings(query)['filings']
# link = fillings[0]['linkToFilingDetails']
# print(f"link: {link}")
# text=__download_form_html(link)
# # Using BeautifulSoup4 as a reliable alternative for HTML parsing
# print(f"step 3 : download html and parse")
# elements = BeautifulSoup(text, 'html.parser').get_text().split('\n')
# elements = [elem.strip() for elem in elements if elem.strip()]  # Clean empty lines and whitespace
# content = "\n".join([str(el) for el in elements])
# #print(f"content: {content}")
# print(f"step 4 : split content into chunks")
# text_splitter = CharacterTextSplitter(
#         separator = "\n",
#         chunk_size = 1000,
#         chunk_overlap  = 150,
#         length_function = len,
#         is_separator_regex = False,
#     )
# docs = text_splitter.create_documents([content])
# print(f"step 5 : create vector store")
# retriever = FAISS.from_documents(
#       docs, OpenAIEmbeddings()
#     ).as_retriever()

# print(f"step 6 : get relevant documents")
# answers = retriever.get_relevant_documents(ask, top_k=4)
# answers = "\n\n".join([a.page_content for a in answers])
# print(f"answers: {answers}")

def get_news(ticker:str)->str:
   """Useful to get news from yfinance for a given ticker"""
   ticker = yf.Ticker(ticker)
   string = []
   for news in ticker.news:
       try:
            news=news['content']
            string.append('\n'.join([ f"Title: {news['title']}", f"Summary: {news['summary']}",f"Published date: {news['pubDate']}", "\n-----------------" ]))
       except KeyError:
            next
   return string 

print(get_news("AAPL"))