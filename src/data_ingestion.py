import sqlite3
import pandas as pd

# Load data from an SQLite database
def load_data(db_path, table_name):
    conn = sqlite3.connect(db_path) # Connect to the database
    query = f"SELECT * FROM {table_name};" # SQL query to select all data
    data = pd.read_sql(query, conn) # Load data into a Pandas DataFrame
    conn.close() # Close the database connection
    return data

if __name__ == "__main__":
    import json # Load configuration settings
    with open("config.json", "r") as f:
        config = json.load(f)
    data = load_data(config["database_path"], config["table_name"])
    data.to_csv("data.csv", index=False)
    print("Data loaded and saved as data.csv")