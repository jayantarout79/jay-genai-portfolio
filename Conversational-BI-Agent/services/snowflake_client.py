import os
import pandas as pd
import snowflake.connector

def get_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ.get("SNOWFLAKE_ROLE", "PUBLIC")
    )

def run_query(sql: str) -> tuple[pd.DataFrame, dict]:
    cnx = get_connection()
    try:
        cur = cnx.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
        df = pd.DataFrame(rows, columns=cols)
        meta = {"rowcount": len(df)}
        return df, meta
    finally:
        cnx.close()