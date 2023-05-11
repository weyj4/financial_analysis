from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import os
import pandas as pd
import psycopg2
import PyPDF2
from sqlalchemy import create_engine
import tiktoken

openai.api_key = os.getenv('OPENAI_API_KEY')

# conn_string = 'postgresql://localhost:5432/postgres'
# db = create_engine(conn_string)
# conn = db.connect()


def convert_pdf_to_text(f):
    text = []
    with open(f, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        # for page in range(len(pdf.pages)):
        for page in range(55,60):
            page_obj = pdf.pages[page]
            text.append(page_obj.extract_text())
    text = "\n".join(text)
    return text

def tiktoken_splitter(chunk_size=300, chunk_overlap=40):
    tokenizer = tiktoken.get_encoding('p50k_base')
    def len_fn(text):
        tokens = tokenizer.encode(
            text, disallowed_special=()
        )
        return len(tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len_fn,
        separators = ["\n\n", "\n", " ", ""]
    )
    return text_splitter

def tokenize_text(t):
    splitter = tiktoken_splitter()
    ds = []

    paper_chunks = splitter.split_text(t)
    for i, chunk in enumerate(paper_chunks):
        ds.append({
            'chunk_id': str(i),
            'chunk': chunk
        })
    return ds

def embed_text(ts):
    for t in ts:
        t['embedding'] = openai.Embedding.create(input=t['chunk'], engine='text-embedding-ada-002')['data'][0]['embedding']
    # return pd.DataFrame(ts)
    return ts

def insert_pg(ts, table_name='finance'):
    conn = psycopg2.connect('dbname=postgres user=weylandjoyner')

    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS finance (id bigserial primary key, chunk text, embedding vector(1536));
    '''
    
    def insert_sql_builder(t):
        t['chunk'] = t['chunk'].replace("\n", " ")
        chunk_id = t['chunk_id']
        chunk = t['chunk']
        embedding = t['embedding']
        print(chunk)
        s = f"INSERT INTO finance (chunk, embedding) VALUES ('{chunk}', '{embedding}');"
        return s

    # df.to_sql(table_name, con=conn, if_exists='replace', index='False')
    cur = conn.cursor()
    cur.execute(create_table_sql)

    for t in ts:
        cur.execute(insert_sql_builder(t))
    conn.commit()
    cur.close()
    conn.close()

def display(res):
    print(res.head())

insert_pg(embed_text(tokenize_text(convert_pdf_to_text('./msft10k.pdf'))))

