import psycopg2
from psycopg2 import OperationalError

def test_connection():
    try:
        conn = psycopg2.connect(
            dbname="mydb",   # <-- anpassen
            user="myuser",       # <-- anpassen
            password="mypassword",  # <-- anpassen
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print("✅ Verbindung erfolgreich!")
        print(f"PostgreSQL-Version: {db_version[0]}")
        cur.execute('''
SELECT table_schema,table_name
FROM information_schema.tables
ORDER BY table_schema,table_name;
                    ''')
        db_version = cur.fetchall()
        print(f"PostgreSQL-Version: {db_version}")
        cur.close()
        conn.close()
    except OperationalError as e:
        print("❌ Fehler bei der Verbindung zur Datenbank:")
        print(e)

if __name__ == "__main__":
    test_connection()
