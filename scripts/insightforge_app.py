######################
## START STREAMLIT ###
######################

#streamlit app Title
import streamlit as st
st.set_page_config(page_title="InsightForge: Business Intelligence Assistant", layout="wide")
st.title("InsightForge: Business Intelligence Assistant")



######################
## LOAD THE DATA #####
######################

import pandas as pd
import numpy as np

sales_data = pd.read_csv("../data/sales_data.csv")



##############################
## CREATING LANGCHAIN SETUP ##
##############################

#define function
#data=sales_data
def generate_advanced_data_summary(dataset=sales_data):

    #open empty summary 
    summary=""
    
    #copy the data
    processed_data=dataset.copy(deep=True)

    #Calculate total sales, average sale, median sale, and standard deviation of sales, 
    #providing a statistical overview of sales performance.
    summary += "## Summary Statistics for the whole sales ##"
    summary += f"\nTotal sales: {str(processed_data['Sales'].sum())}"
    summary += f"\nMean: {str(processed_data['Sales'].mean())}"
    summary += f"\nMedian: {str(processed_data['Sales'].mean())}"
    summary += f"\nStandard deviation: {str(processed_data['Sales'].std())}"

    #Aggregates sales data by month and identifies the best and worst
    #performing months based on sales volume.
    processed_data["Date"]=pd.to_datetime(processed_data["Date"])
        #Convert the 'Date' column to datetime format to enable time-based analysis
    processed_data['month'] = processed_data['Date'].dt.strftime('%B')
    #processed_data["month"] = processed_data["Date"].dt.month
    monthly_median = processed_data.groupby("month")["Sales"].median().reset_index()
    best_month=monthly_median.loc[monthly_median["Sales"]==monthly_median["Sales"].max(), "month"].to_list()
    worst_month=monthly_median.loc[monthly_median["Sales"]==monthly_median["Sales"].min(), "month"].to_list()
    summary += f"\n\n## Median sales per month ##"
    summary += f"\n{monthly_median}"

    #Analyze sales data by product, identifying the top-selling product by total sales 
    #value and the most frequently sold product by sales count.
    product_median = processed_data.groupby("Product")["Sales"].median().reset_index()
    product_count = processed_data.groupby("Product").size().reset_index(name='count')
    best_product_volume=product_median.loc[product_median["Sales"]==product_median["Sales"].max(), "Product"].to_list()
    best_product_freq=product_count.loc[product_count["count"]==product_count["count"].max(), "Product"].to_list()
    summary += f"\n\n## Median of sale volume per product ##"
    summary += f"\n{product_median}"
    summary += f"\n\n## Sales count per product ##"
    summary += f"\n{product_count}"

    #Aggregates sales data by region, identifying the best and worst performing regions
    region_median = processed_data.groupby("Region")["Sales"].median().reset_index()
    best_region=region_median.loc[region_median["Sales"]==region_median["Sales"].max(), "Region"].to_list()
    worst_region=region_median.loc[region_median["Sales"]==region_median["Sales"].min(), "Region"].to_list()
    summary += f"\n\n## Sales per region ##"
    summary += f"\n {region_median}"

    #Analyze customer satisfaction scores mean and standard deviation.
    summary += "\n\n## Customer satisfaction statistics: "
    summary += f"\nMedian: {str(processed_data['Customer_Satisfaction'].mean())}; "
    summary += f"\nStandard deviation: {str(processed_data['Customer_Satisfaction'].std())}"
    
    #Segment customers by age group and calculates average sales for each group, 
    #identifying the best-performing age group.
    bins = [18, 30, 40, 50, 60, 70]
    labels = ["18_30", "30_40", "40_50", "50_60", "60_70"]
    processed_data["age_group"] = pd.cut(processed_data["Customer_Age"], bins=bins, labels=labels, right=False)
    age_median=processed_data.groupby("age_group", observed=True)["Sales"].median().reset_index()
        #The observed argument in the groupby method in pandas is used to control 
        #whether or not to include only the observed groups in the result 
        #when grouping by a categorical variable.
        #True: When observed is set to True, the result will include only 
        #the groups that are actually observed in the data. 
    best_age = age_median.loc[age_median["Sales"]==age_median["Sales"].max(), "age_group"].to_list()
    worst_age = age_median.loc[age_median["Sales"]==age_median["Sales"].min(), "age_group"].to_list()
    summary += f"\n\n## Average sales per age group ##"
    summary += f"\n {age_median}"
    
    #Analyze average sales by customer gender.
    gender_median = processed_data.groupby("Customer_Gender")["Sales"].median().reset_index()
    best_gender = gender_median.loc[gender_median["Sales"]==gender_median["Sales"].max(), "Customer_Gender"].to_list()
    worst_gender = gender_median.loc[gender_median["Sales"]==gender_median["Sales"].min(), "Customer_Gender"].to_list()
    summary += f"\n\n## Median sales per gender ##"
    summary += f"\n {gender_median}"
    
    #add key point
    summary += f"""
        \nKey points for the sales of our business:
            \nOur total sales was {str(processed_data['Sales'].sum())}
            \nOur average customer satisfaction was {str(processed_data['Customer_Satisfaction'].mean())}
            \nThe month with the higest sales was '{best_month[0]}', while the one with the least was {worst_month[0]}
            \nThe product category with the higest sales was '{best_product_volume[0]}'
            \nThe region with the higest sales was '{best_region[0]}' while the one with the least was {worst_region[0]}
            \nThe age group with the higest sales was '{best_age[0]}' while the one with the last was {worst_age[0]}
            \nThe gender with the higest sales was '{best_gender[0]}' while the one with the least was {worst_gender[0]}
    """
    
    #return the summary
    return summary
    
#run the function
advanced_summary = generate_advanced_data_summary()

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    #initializes a language model using the ChatOpenAI class 
        #with the specified model name (gpt-3.5-turbo)
    #The temperature parameter controls the randomness of 
        #the model's output. A temperature of 0 makes the 
        #model's responses more deterministic and focused.
    #For this project, setting the temperature very low (e.g., 0.3) would make 
        #the agent unable to do some tasks like extracting statistical
        #information from text summaries we previously created



###################
## REGULAR AGENT ##
###################

#define template
scenario_template = """
    You are an expert AI sales analyst. Please answer the following question:
    {question}
"""

#set prompt without RAG
from langchain import PromptTemplate
prompt = PromptTemplate(
    input_variables=["question"],
    template=scenario_template
)

#define the agent
from langchain.chains import LLMChain
regular_agent = LLMChain(prompt=prompt, llm=llm)



###########################
##KNOLEDGE BASE CREATION ##
###########################

# list all the PDFs in the PDF Folder
import os
import fnmatch
path_pdfs="../data/pdf_folder/"
list_pdfs=fnmatch.filter(os.listdir(path_pdfs), "*.pdf") 

#load them
from langchain.document_loaders import PyPDFLoader
#pdf="RIL_IAR_2024.pdf"
extracted_list = list()
for pdf in list_pdfs:
    final_path=path_pdfs+pdf
    print(final_path)
    if(os.path.exists(final_path)):
        Doc_loader = PyPDFLoader(final_path)
        extracted_text=Doc_loader.load()
        extracted_list.append(extracted_text)
        #the result is a list of lists, where each list include the text of each PDF

#split chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
    #each chunk will have a maximum of 150 characters
    #no character will overlap between chunks
    #Multiple separators to split the text. It first tries to split the text at the first separator, if it cannot split the text without exceeding the chunk_size, it will move to th enext separator and so on...
        #"\n\n": Double newline, often used to separate paragraphs.
        #"\n": Single newline, often used to separate lines.
        #"(?<=\. )": A regular expression that matches a period followed by a space, often used to separate sentences.
            #It asserts that what immediately precedes the current position in the text must match the pattern inside the parentheses.
            #\. matches a literal period (dot) character. The backslash \ is used to escape the dot, which is a special character in regular expressions that normally matches any character.
            #The space character matches a literal space
            #Putting it all together, (?<=\. ) matches a position in the text that is immediately preceded by a period followed by a space. 
        #" ": A space character, used to separate words.
        #"": An empty string, which means that if no other separators work, the text will be split at any character to ensure the chunk size is respected.

#make a list with all the splits
split_list=list()
for index, text in enumerate(extracted_list):
    
    #start
    print(f"\n##### Starting PDF number {index} #######")
    
    #split the corresponding text
    split_text=text_splitter.split_documents(text)
    
    #print the length of the chunks and a the first chunk as an example
    print(f"## The number of chunks is {len(split_text)} ##")
    print("## First chunk as an example ##")
    print(split_text[0])
    
    #save in a list
    split_list.append(split_text)

#save
import pickle
with open("../data/pdfs_chunks.pkl", 'wb') as file:
    pickle.dump(split_list, file)

#To load it
#with open("../data/pdfs_chunks.pkl", 'rb') as file:
    #split_list = pickle.load(file)



####################
## SETTING UP RAG ##
####################

# Load processed texts from pickle file
import pickle
with open("../data/pdfs_chunks.pkl", 'rb') as file:
    split_list = pickle.load(file)

# Flatten the list of lists into a single list of chunks
flat_documents = [chunk for sublist in split_list for chunk in sublist]
print(flat_documents[0:2])

#create embeddings
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
    #Now create the embeddings. An embedding is a numerical representation of data, 
    #typically in the form of a vector, that captures the semantic meaning or features 
    #of the data in a lower-dimensional space. Embeddings reduce the dimensionality of 
    #the data while preserving its essential features. Embeddings capture semantic 
    #relationships between data points. For example, in word embeddings, words with 
    #similar meanings are represented by vectors that are close to each other in the 
    #embedding space.
    #Example: The words "king" and "queen" might have similar embeddings because 
    #they are semantically related.
    #The difference between the embeddings of "king" and "queen" might be similar 
    #to the difference between the embeddings of "man" and "woman".

# Create the FAISS vector store
from langchain.vectorstores import FAISS
vector_store = FAISS.from_documents(flat_documents, embeddings)
    #FAISS (Facebook AI Similarity Search) is a library for efficient 
    #similarity search and clustering of dense vectors.



#############################
## CREATE AGENT WITH TOOLS ##
#############################

#Define the RetrievalQA chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type( \
    llm=llm, \
    retriever=vector_store.as_retriever(), \
    return_source_documents=True \
)
    #This code defines the RetrievalQA chain using the from_chain_type method.
    #The from_chain_type method is a convenient way to create a RetrievalQA 
    #chain with specific parameters.
    #retriever=vector_store.as_retriever():
        #This parameter specifies the retriever to be used in the chain.
        #vector_store.as_retriever() converts the FAISS vector store into
        #a retriever that can be used to fetch relevant documents based on
        #similarity search.
    #return_source_documents=True:
        #This parameter ensures that the sources of the information (i.e., 
        #the documents retrieved by the retriever) are returned along with the response.
        #Setting this parameter to True allows you to see which documents were used to 
        #generate the response.

#wikipedia 
#!pip install wikipedia
import wikipedia

#define function to search in wikipedia
def wiki_search(query):
    
    #The try block is used to handle exceptions that might 
    #occur during the execution of the code within it.
    try:
        
        #Search Wikipedia for the query
        search_results = wikipedia.search(query)
        
        #If the search results are empty, the function 
        #returns "No results found."
        if not search_results:
            return "No results found."

        #fetch the corresponding Wikipedia page, 
        top_result = search_results[0]
        
        #extract a summary of the page (limited to 3 sentences), 
        page = wikipedia.page(top_result)

        #and get the URL of the page.
        summary = wikipedia.summary(top_result, sentences=3)
        
        #get URL
        url = page.url

        #return only the summary and the URL
        return {"summary": summary, "url": url}

    #Handle Disambiguation Errors
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
        #If a disambiguation error occurs (i.e., the query is 
        #ambiguous and could refer to multiple pages), this 
        #block catches the exception and returns a message 
        #with the possible options.
    except wikipedia.PageError:
        return "Page not found."
        #If a page error occurs (i.e., the page does not exist), 
        #this block catches the exception and returns "Page not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"
        #If any other exception occurs, this block catches 
        #the exception and returns a message with the error details.

#run example
#wiki_search("America")

#create wikipeda search tool
from langchain.tools import Tool

#This block defines a class named WikipediaAPIWrapper. 
class WikipediaAPIWrapper:
    def __init__(self):
        pass

    def search(self, query):
        return wiki_search(query)
    #This class serves as a wrapper around the wiki_search function. 
    #The class has an __init__ method, which is a constructor that
    #initializes the class instance. The search method takes a query 
    #as an argument and returns the result of the wiki_search function.

#create the Wikipedia search tool
wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper().search,
    description="Searches Wikipedia for a given query and returns the summary and URL of the top result."
)
    #This block creates an instance of the Tool class named wikipedia_tool. 
    #The Tool class is initialized with the following parameters:
        #name: A string that specifies the name of the tool. In this case, 
            #it is "Wikipedia Search".
        #func: The function that the tool will use to perform its task. 
            #Here, it is set to the search method of the WikipediaAPIWrapper class.
        #description: A string that provides a description of what the tool does. 
            #In this case, it describes that the tool searches Wikipedia for a 
            #given query and returns the summary and URL of the top result.

#Define generate_rag_insight
#This function will combine the retrieved documents and Wikipedia content, 
def generate_rag_insight(question):
    
    #use the qa_chain with the retriever (FAISS vector storage) to get
    #answers based on the PDFs
    retrieved_docs = qa_chain({"query": question})
        #we will get from here the documents relevant for the input
        #question (see below)
    
    #Query Wikipedia for additional content
    wiki_response = wikipedia_tool.func(question)

    # Step 3: Combine the retrieved documents and Wikipedia content into a single context
    combined_context = f"""
    
        You are assisting an AI business analyst by providing recommendations considering as context the following relevant documents about sales and marketing: 
        {retrieved_docs['source_documents']}\n
        
        And a related wikipedia search:
        {wiki_response['summary']}\n
        URL: {wiki_response['url']}
        
        Considering all this, please provide specific recommendations tailored to the following question:
        {question}
    """

    #Use the qa_chain to generate the final insight based on the combined context
    final_insight = qa_chain({"query": combined_context})

    #Compile the sources (retrieved documents and Wikipedia URLs)
    sources = {
        "retrieved_documents": [doc.metadata["source"] for doc in retrieved_docs["source_documents"]],
        "wikipedia_url": wiki_response["url"]
    }

    # Step 6: Return the final insight and sources
    return {
        "insight": final_insight,
        "sources": sources
    }

#run example
#generate_rag_insight("what are relevant factors for the sales in our specific business and why")

#plot sales per product category
import matplotlib.pyplot as plt

def plot_product_category_sales(dataset=sales_data):
    
    #group by product and calculate total sales
    product_sales = dataset.groupby('Product')['Sales'].sum().sort_values(ascending=False)

    #create the bar plot
    plt.figure(figsize=(20, 12))
    product_sales.plot(kind='bar', color='skyblue')
    plt.title('Sales Distribution by Product')
    plt.xlabel('Product')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    #save the plot as a PNG file
    file_path = "../data/product_sales_distribution.jpeg"
    plt.savefig(file_path)
    plt.close()

    return file_path
      #this is relevant as the path will be used by streamlit to show the figures in the app

#plot_product_category_sales()

#plot of sales per year
import matplotlib.pyplot as plt

def plot_yearly_sales_trend(dataset=sales_data):

    #convert the 'Date' column to datetime if not already
    dataset['Date'] = pd.to_datetime(dataset['Date'])

    #extract year and month from the 'Date' column
    dataset['Year'] = dataset['Date'].dt.year
    year_names = sales_data["Year"].unique()
    
    #group by year and month, and calculate total sales
    yearly_sales = dataset.groupby(['Year'])['Sales'].sum().reset_index()

    #min and max sales per month
    min_sales = yearly_sales["Sales"].min()
    max_sales = yearly_sales["Sales"].max()

    #create the bar plot
    plt.figure(figsize=(20, 12))
    yearly_sales.plot(kind='bar', color='skyblue')
    plt.title('Sales Distribution by Year')
    plt.ylim(min_sales-(min_sales*0.10), max_sales+(max_sales*0.10))
    plt.xticks(ticks=range(0, len(year_names)), labels=year_names)
    
    plt.xlabel('Year')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    #save the plot as a PNG file
    file_path = "../data/yearly_sales_trend.jpeg"
    plt.savefig(file_path)
    plt.close()

    return file_path
      #this is relevant as the path will be used by streamlit to show the figures in the app

#plot_yearly_sales_trend()

#add the tools
from langchain.tools import Tool
#each of these tool will take a callable function as an argument (func)
#For example, "product_category_sales_tool" will call the function
#"plot_product_category_sales" to create a plot of the sales per product
#this is the case for all except for advanced_summary_tool, where
#we directly provide the summary previously created with the
#corresponding function

#you also have to add a name a description 
#These descriptions are VERY IMPORTANT as they are added to prompt
#and help the agent to decide about what is the best tool to use
#given the input question

#tool for advance summary of the data
advanced_summary_tool = Tool(
    name="AdvancedSummary",
    func=lambda x: advanced_summary,
    description="Provides an advanced data analysis of our sales data in our business. You can use this to obtain average values for different products, regions, age groups, etc..."
)

# Tool for Product Category Sales Plot
product_category_sales_tool = Tool(
    name="ProductCategorySalesPlot",
    func=lambda x: plot_product_category_sales,
    description="Generates a bar plot showing sales distribution by product category."
)

# Tool for Sales Trend Plot
sales_trend_tool = Tool(
    name="SalesTrendPlot",
    func=lambda x: plot_yearly_sales_trend,
    description="Generates a line plot showing the trend of sales across years."
)

# Tool for RAG Insight
rag_insight_tool = Tool(
    name="RAGInsight",
    func=lambda x: generate_rag_insight,
    description="Generates insights using the RAG system (using documents about sales and marketing) and external knowledge (based on wikipedia)."
)

#you can get tools using langchain.agents.load_tools(["llm-math", "wikipedia"], llm=llm)


# Define the prefix for the agent's prompt
from langchain.agents import ZeroShotAgent
prefix = """
    You are an expert AI sales analyst. You can analyze sales data, generate visualizations, 
    and provide insights based on internal data and external knowledge.
    
    Always explain your reasoning step by step before providing the final answer.

    You have access to the following tools:
"""
    #after the prefix and before the prefix, "ZeroShotAgent.create_prompt" will list
    #all available tools

# Define suffix for the agent's prompt
suffix = """
    Start
    
    {chat_history}
    
    User: {input}
    
    Agent: We are going to approach this step by step: {agent_scratchpad}
"""
    #Start: {chat_history}
        #This placeholder ({chat_history}) represents the conversation history between 
            #the user and the agent so far.
        #It allows the agent to maintain context across multiple interactions. For example, 
            #if the user asks follow-up questions, the agent can refer back to previous exchanges.
        #When the agent is invoked, the system dynamically replaces {chat_history} with 
            #the actual conversation history (e.g., previous user inputs and agent responses)
    #User: {input}
        #This placeholder ({input}) represents the current question or input from the user.
        #It tells the agent what the user is asking or requesting in this specific interaction.
        #When the agent is invoked, the system dynamically replaces {input} with the user's current query.
    #Agent: We are going to approach this step by step: {agent_scratchpad}
        #This is where the agent starts its response.
        #The agent begins by explaining its reasoning step by step, 
            #as instructed in the prefix.
        #The placeholder {agent_scratchpad} is where the agent 
            #"thinks out loud" or writes down its intermediate reasoning 
            #and steps before providing the final answer.
            #This makes the agent's reasoning transparent and easier to follow.
            
# Create the prompt using ZeroShotAgent
tools = [advanced_summary_tool, product_category_sales_tool, sales_trend_tool, rag_insight_tool]
prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["chat_history", "input", "agent_scratchpad"]
)

#take a look to the prompt
#prompt


from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor

# Create the LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Create the ZeroShotAgent
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    #The verbose=True parameter in both the ZeroShotAgent and 
        #AgentExecutor functions controls the level of logging and 
        #output detail during their execution. Here's what it means 
        #in each context
    #The handle_parsing_errors=True parameter in the AgentExecutor ensures 
        #that the agent can gracefully handle parsing errors that occur 
        #during the execution of the agent's reasoning or response generation




################
## MONITORING ##
################

import json
import time
import matplotlib.pyplot as plt
from datetime import datetime

#define a class encapsulates all the functionality for monitoring and logging model performance
class SimpleModelMonitor:
    
    #Initializes the SimpleModelMonitor instance.
    def __init__(self, log_file="../data/model_logs.json"):
        self.log_file = log_file
        self.logs = self.load_logs()
        #Sets the default log file name (log_file) where logs will be saved.
        #Loads existing logs from the file using the load_logs method
            #defined below, this will go to "logs" which are the current logs

    #load logs from a JSON file
    def load_logs(self):
        try:
            with open(self.log_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []  # Return an empty list if the log file doesn't exist
        #Reads logs from the specified JSON file (log_file).
            #Uses json.load to parse the file contents into a Python list.
        #If the file does not exist, it returns an empty list.
            #so it ensures the program doesn't crash if the file is missing.

    #save logs to a JSON file
    def save_logs(self):
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=4)
        #Saves the current logs (self.logs) to the specified JSON file (log_file).
            
    #log an interaction with the model
    def log_interaction(self, query, execution_time):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "execution_time": execution_time
        }
        self.logs.append(log_entry)
        self.save_logs()
        #Logs an interaction with the model, including the query, execution time, and a timestamp.
        #Creates a dictionary (log_entry) with:
            #timestamp: The current date and time in ISO 8601 format.
            #query: The query sent to the model.
            #execution_time: The time taken to process the query.
        #Appends the log entry to the self.logs list.
        #Calls save_logs to save the updated logs to the JSON file.
            #This function was defined above

    #get the average execution time across all logged interactions
    def get_average_execution_time(self):
        if not self.logs:
            return 0
        total_time = sum(log["execution_time"] for log in self.logs)
        return total_time / len(self.logs)
        #Checks if there are any logs. If not, returns 0.
        #Uses a generator expression to get the execution_time values from all logs.
            #and then sum them all
        #Divides the total execution time by the number of logs to compute the average.

    #plot execution times over time
    def plot_execution_times(self):
        
        #if no logs, nothing to plot
        if not self.logs:
            print("No logs available to plot.")
            return

        #get the timestamps and execution times of ALL logs
        timestamps = [log["timestamp"] for log in self.logs]
        execution_times = [log["execution_time"] for log in self.logs]

        #plot timestamp against execution time to see the variation in 
            #execution time
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, execution_times, marker="o", linestyle="-", color="blue")
        plt.title("Model Execution Times Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Execution Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        #save the plot as a PNG file
        file_path = "../data/execution_times.jpeg"
        plt.savefig(file_path)
        plt.close()

        return file_path



################
## EVALUATION ##
################

#questions/answers
qa_pairs = [
    {
        "question": "What are the total sales?",
        "answer": f"The total sales amount is ${sales_data['Sales'].sum():,.2f}."
    },
    {
        "question": "What was the region with the highest sales?",
        "answer": f"The region with the highest sales is {sales_data.groupby('Region')['Sales'].sum().idxmax()}."
    },
    {
        "question": "What was the gender with the highest sales?",
        "answer": f"The gender with the highest sales is {sales_data.groupby('Customer_Gender')['Sales'].sum().idxmax()}."
    },
    {
        "question": "What was the gender with the lowest sales?",
        "answer": f"The gender with the lowest sales is {sales_data.groupby('Customer_Gender')['Sales'].sum().idxmin()}."
    },
]

#function to evaluate
from langchain.evaluation.qa import QAEvalChain
def evaluate_model():

    #create the evaluation chain
    eval_chain = QAEvalChain.from_llm(llm, handle_parsing_errors=True)

    #generate predictions for each question
    predictions = []
    
    #loop across question/answer pairs
    #qa_pair=qa_pairs[0]
    for qa_pair in qa_pairs:
        
        #start timing
        start_time = time.time()
        
        #get question
        question = qa_pair["question"]
        
        #get answer
        try:
            
            #run the question through the agent to get the prediction
            prediction = agent_executor.run(input=question, chat_history="", agent_scratchpad="")
            
            #save the prediction
            predictions.append({"question": question, "prediction": prediction})
        except Exception as e:
            # Handle errors gracefully
            predictions.append({"question": question, "prediction": f"Error: {str(e)}"})

        #end timing
        end_time = time.time()
        execution_time = end_time - start_time

        #log the interaction using the corresponding function for the model_monitor class
        model_monitor.log_interaction(question, execution_time)


    #evaluate predictions against actual answers
    evaluation_results = []
    for qa_pair, prediction in zip(qa_pairs, predictions):

        result = eval_chain.evaluate(
            examples=[qa_pair],  # Evaluate one pair at a time
            predictions=[prediction],
            question_key="question",
            answer_key="answer",
            prediction_key="prediction"
        )

        evaluation_results.append({ \
            "question": qa_pair["question"], \
            "answer": qa_pair["answer"], \
            "prediction": prediction["prediction"], \
            "result": result[0]["results"] \
        })

    return evaluation_results

#run evaluation
#evaluation_results = evaluate_model()



###############
## STREAMLIT ##
###############

import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import time

#initialize the model monitor
import os
os.system("rm ../data/model_logs.json")
  #remove json file if exists
model_monitor = SimpleModelMonitor(log_file="../data/model_logs.json")
  #Calls the __init__ method of the SimpleModelMonitor class.
  #Passes the argument log_file="../data/model_logs.json" to the __init__ method.
  #Executes the code inside the __init__ method to initialize the instance.

#sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "AI Assistant", "Model Performance"])

#home Page
if page == "Home":
    st.header("Welcome to InsightForge!")
    st.write("""
        InsightForge is your AI-powered Business Intelligence Assistant. 
        Use the sidebar to navigate through the app and explore the following features:
        - **Data Analysis**: View sales summaries and trends.
        - **AI Assistant**: Interact with the AI assistant for insights.
        - **Model Performance**: Monitor and evaluate the AI model's performance.
    """)

#data Analysis Page
elif page == "Data Analysis":
    st.header("Data Analysis")

    #sales Summary
    st.subheader("Sales Summary")
    advanced_summary = f"""
    - Total Sales: ${sales_data['Sales'].sum():,.2f}
    - Total Products: {sales_data['Product'].nunique()}
    - Total Days: {sales_data['Date'].nunique()}
    """
    st.text(advanced_summary)

    #sales Distribution by Product Category
    st.subheader("Sales Distribution by Product Category")
    product_sales_plot = plot_product_category_sales(sales_data)
    st.image(product_sales_plot, caption="Sales Distribution by Product Category")
      #This function makes the figure and returns the path where this figure is located
      #then this is used by st.image to load the figure

    #yearly sales Trend
    st.subheader("Yearly Sales Trend")
    sales_trend_plot = plot_yearly_sales_trend(sales_data)
    st.image(sales_trend_plot, caption="Yearly Sales Trend")

# AI Assistant Page
elif page == "AI Assistant":
    st.header("AI Assistant")

    #mode Selection
    mode = st.radio("Choose Assistant Mode", ["Standard", "RAG"])

    #user Input
    user_query = st.text_input("Ask a question:")

    #if press submit
    if st.button("Submit"):
        if user_query:
            
            #start timing
            start_time = time.time()

            #run the agent
            if mode == "Standard":
                #just LLM without RAG
                response = regular_agent.run(user_query)
            elif mode == "RAG":
                #LLM with RAG (PDFs, wikipedia and our sales data)
                response = agent_executor.run(input=user_query, chat_history="", agent_scratchpad="")

            #end timing
            end_time = time.time()
            execution_time = end_time - start_time

            #display response and execution time
            st.subheader("Response")
            st.write(response)
            st.write(f"Execution Time: {execution_time:.2f} seconds")

            #log the interaction using the corresponding function for the model_monitor class
            model_monitor.log_interaction(user_query, execution_time)

#model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")

    #model Evaluation
    st.subheader("Model Evaluation")

    #run evaluation using the agent with monitor
    evaluation_results = evaluate_model()

    #save the predictions
    predictions = [result["prediction"] for result in evaluation_results]
    results = [result["result"] for result in evaluation_results]

    #bind questions, answers and predictions
    evaluation_results = [
      {"question": qa["question"], "prediction": pred, "answer": qa["answer"], "correct": res}
      for qa, pred, res in zip(qa_pairs, predictions, results)
    ]
      #combine the questions/answer pairs used in evaluation plus the predictions and the result of the 
      #evaluation

    #display Evaluation Results
    for result in evaluation_results:
        st.write(f"**Question**: {result['question']}")
        st.write(f"**Predicted Answer**: {result['prediction']}")
        st.write(f"**Actual Answer**: {result['answer']}")
        st.write(f"**Correct**: {result['correct']}")
        st.write("---")

    #execution Time Monitoring
    st.subheader("Execution Time Monitoring")
      #we will use the functions previously defined in the model_monitor class

    #plot
    plot_execution = model_monitor.plot_execution_times()
    st.image(plot_execution, caption="Execution time")
    
    #average
    avg_time = model_monitor.get_average_execution_time()
    st.write(f"**Average Execution Time**: {avg_time:.2f} seconds")
