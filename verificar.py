from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
load_dotenv()

engine = create_engine(f'postgresql+psycopg2://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}/{os.getenv("DB_NAME")}?sslmode=require')

with engine.connect() as conn:
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'eoq_resultados'"))
    print([r[0] for r in result])