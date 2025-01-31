from langchain.tools import tool
from dotenv import load_dotenv
import os,json,requests
from sec_api import QueryApi
from dotenv import load_dotenv
import yfinance as yf

from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

from crewai import Agent,Crew
from crewai import Task
from textwrap import dedent

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Get news directly using yfinance


# ######## tools #########################
class CalculatorTools():

  @tool("Make a calculation")
  def calculate(operation):
    """Useful to perform any mathematical calculations, 
    like sum, minus, multiplication, division, etc.
    The input to this tool should be a mathematical 
    expression, a couple examples are `200*7` or `5000/2*10`
    """
    return eval(operation)


class SearchTools():
    
  @tool("Get news from yfinance")
  def yahoo_finance_news(ticker:str)->str:
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
    
    
    
  @tool("Search internet")
  def search_internet(query:str)->str:
    """Useful to search the internet 
    about a a given topic and return relevant results"""
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([ f"Title: {result['title']}", f"Link: {result['link']}",f"Snippet: {result['snippet']}", "\n-----------------" ]))
      except KeyError:
        next

    return '\n'.join(string)

  @tool("Search news on the internet")
  def search_news(query:str)->str:
    """Useful to search news about a company, stock or any other topic and return relevant results"""""
    top_result_to_return = 4
    url = "https://google.serper.dev/news"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['news']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([f"Title: {result['title']}", f"Link: {result['link']}", f"Snippet: {result['snippet']}", "\n-----------------"]))
      except KeyError:
        next

    return '\n'.join(string)
#


class SECTools():
  @tool("Search 10-Q form")
  def search_10q(data:str)->str:
    """
    Useful to search information from the latest 10-Q form for a
    given stock.
    The input to this tool should be a pipe (|) separated text of
    length two, representing the stock ticker you are interested and what
    question you have from it.
		For example, `AAPL|what was last quarter's revenue`.
    """
    stock, ask = data.split("|")
    queryApi = QueryApi(api_key=os.environ['SEC_API_KEY'])
    query = {
      "query": {
        "query_string": {
          "query": f"ticker:{stock} AND formType:\"10-Q\""
        }
      },
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']
    if len(fillings) == 0:
      return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = SECTools.__embedding_search(link, ask)
    return answer

  @tool("Search 10-K form")
  def search_10k(data):
    """
    Useful to search information from the latest 10-K form for a
    given stock.
    The input to this tool should be a pipe (|) separated text of
    length two, representing the stock ticker you are interested, what
    question you have from it.
    For example, `AAPL|what was last year's revenue`.
    """
    stock, ask = data.split("|")
    queryApi = QueryApi(api_key=os.environ['SEC_API_KEY'])
    query = {
      "query": {
        "query_string": {
          "query": f"ticker:{stock} AND formType:\"10-K\""
        }
      },
      "from": "0",
      "size": "1",
      "sort": [{ "filedAt": { "order": "desc" }}]
    }

    fillings = queryApi.get_filings(query)['filings']
    if len(fillings) == 0:
      return "Sorry, I couldn't find any filling for this stock, check if the ticker is correct."
    link = fillings[0]['linkToFilingDetails']
    answer = SECTools.__embedding_search(link, ask)
    return answer

  def __embedding_search(url:str, ask:str)->str:
    text = SECTools.__download_form_html(url)
    elements = BeautifulSoup(text, 'html.parser').get_text().split('\n')
    elements = [elem.strip() for elem in elements if elem.strip()]  # Clean empty lines and whitespace
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 150,
        length_function = len,
        is_separator_regex = False,
    )
    docs = text_splitter.create_documents([content])
    retriever = FAISS.from_documents( docs, OpenAIEmbeddings() ).as_retriever()
    answers = retriever.get_relevant_documents(ask, top_k=4)
    answers = "\n\n".join([a.page_content for a in answers])
    return answers

  def __download_form_html(url:str)->str:
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

######## tools #########################

######## agents #########################


class StockAnalysisAgents():
    def __init__(self):
        self.llm = ChatOpenAI(
           model="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    def financial_analyst(self):
        return Agent(
            role='The Best Financial Analyst',
            goal="""Impress all customers with your financial data and market trends analysis""",
            backstory="""The most seasoned financial analyst with lots of expertise in stock market analysis and investment strategies that is working for a super important customer.""",
            verbose=True,
      tools=[
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ],
      llm=self.llm,
      allow_delegation=True,
    )
        
    def research_analyst(self):
        return Agent(
            role='Staff Research Analyst',
            goal="""Being the best at gather, interpret data and amaze your customer with it""",
            backstory="""Known as the BEST research analyst, you're skilled in sifting through news, company announcements, and market sentiments. Now you're working on a super important customer""",
      verbose=True,
      tools=[
        SearchTools.search_internet,
        SearchTools.search_news,
        SearchTools.yahoo_finance_news,
        SECTools.search_10q,
        SECTools.search_10k
      ],
      llm=self.llm,
      allow_delegation=True,
  )
    def investment_advisor(self):
        return Agent(
            role='Private Investment Advisor',
            goal="""Impress your customers with full analyses over stocks and complete investment recommendations""",
            backstory="""You're the most experienced investment advisor and you combine various analytical insights to formulate strategic investment advice. You are now working for a super important customer you need to impress.""",
            verbose=True,
      tools=[
        SearchTools.search_internet,
        SearchTools.search_news,
        CalculatorTools.calculate,
        SearchTools.yahoo_finance_news
      ],
      llm=self.llm,
      allow_delegation=True,
    )
        
# #######   task ###########

class StockAnalysisTasks():
    def research(self, agent, company):
        return Task(description=dedent(f"""
        Collect and summarize recent news articles, press releases, and market analyses related to the stock and its industry.
        Pay special attention to any significant events, market sentiments, and analysts' opinions. Also include upcoming events like earnings and others.
        Your final answer MUST be a report that includes a comprehensive summary of the latest news, any notable shifts in market sentiment, and potential impacts on the stock.
        Also make sure to return the stock ticker.
        {self.__tip_section()}
        Make sure to use the most recent data as possible.
        Selected company by the customer: {company}
      """),
      expected_output="A comprehensive research report with news summary, market sentiment analysis, and stock ticker",
      agent=agent
    )
    
    def financial_analysis(self, agent, company): 
        return Task(description=dedent(f"""
        Conduct a thorough analysis of the {company} stock's financial health and market performance. 
        This includes examining key financial metrics such as P/E ratio, EPS growth, revenue trends, and  debt-to-equity ratio. 
        Also, analyze the stock's performance in comparison to its industry peers and overall market trends.

        Your final report MUST expand on the summary provided but now including a clear assessment of the stock's financial standing, its strengths and weaknesses, and how it fares against its competitors in the current market scenario.{self.__tip_section()}

        Make sure to use the most recent data possible.
      """),
      expected_output="A detailed financial analysis report with key metrics and comparative analysis",
      agent=agent,
    )

    def filings_analysis(self, agent, company):
        return Task(description=dedent(f"""
        Analyze the latest 10-Q and 10-K filings from EDGAR for the {company} stock in question. 
        Focus on key sections like Management's Discussion and Analysis, financial statements, insider trading activity, and any disclosed risks.
        Extract relevant data and insights that could influence the stock's future performance.
        Your final answer must be an expanded report that now also highlights significant findings from these filings, including any red flags or positive indicators for your customer.
        {self.__tip_section()}        
      """),
      expected_output="A comprehensive SEC filings analysis report with key findings and risk assessment",
      agent=agent
    )
    
    def recommend(self, agent, company):
        return Task(description=dedent(f"""
        Review and synthesize the analyses provided by the Financial Analyst and the Research Analyst.
        Combine these insights to form a comprehensive investment recommendation for {company}. 
        
        You MUST Consider all aspects, including financial health, market sentiment, and qualitative data from EDGAR filings.

        Make sure to include a section that shows insider trading activity, and upcoming events like earnings.

        Your final answer MUST be a recommendation for your customer. 
        It should be a full super detailed report, providing a clear investment stance and strategy with supporting evidence.
        Make it pretty and well formatted for your customer.
        {self.__tip_section()}
      """),
      expected_output="A detailed investment recommendation report with clear strategy and supporting evidence",
      agent=agent
    )
    
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
    
#     ############ main ############
    
    
company = "AAPL"
    
agents = StockAnalysisAgents()
tasks = StockAnalysisTasks()

research_analyst_agent = agents.research_analyst()
financial_analyst_agent = agents.financial_analyst()
investment_advisor_agent = agents.investment_advisor()

research_task = tasks.research(research_analyst_agent, company)
financial_task = tasks.financial_analysis(financial_analyst_agent, company)
filings_task = tasks.filings_analysis(financial_analyst_agent, company)
recommend_task = tasks.recommend(investment_advisor_agent, company)

crew = Crew(
      agents=[
        research_analyst_agent,
        financial_analyst_agent,
        investment_advisor_agent
      ],
      tasks=[
        research_task,
        financial_task,
        filings_task,
        recommend_task
      ],
      verbose=True
    )

result = crew.kickoff()
print(f"result: {result}")