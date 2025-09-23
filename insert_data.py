import os
import re
import psycopg2
import pandas as pd
from io import StringIO
from sentence_transformers import SentenceTransformer

# --- DB Verbindung ---
conn = psycopg2.connect(
    dbname="mydb",
    user="myuser",
    password="mypassword",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# --- Embedding Modell ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384  # Modellabhängig

# --- Hilfsfunktionen ---
def create_embedding(text: str):
    """Erzeuge Embedding mit lokalem Modell"""
    return embedder.encode(text).tolist()

def insert_entity(title, category, source, content):
    cur.execute(
        "INSERT INTO entities (title, category, source, content) VALUES (%s, %s, %s, %s) RETURNING id",
        (title, category, source, content)
    )
    return cur.fetchone()[0]

def insert_embedding(entity_id, embedding):
    cur.execute(
        "INSERT INTO embeddings (entity_id, embedding) VALUES (%s, %s)",
        (entity_id, embedding)
    )

# --- Parser für Markdown Tabellen ---
def extract_markdown_tables(md_text):
    """Finde Markdown Tabellen und wandle sie in Pandas DataFrames um"""
    tables = []
    matches = re.findall(r"(?:\|.+\|[\r\n]+)+", md_text)
    for table_md in matches:
        try:
            df = pd.read_csv(StringIO(table_md), sep="|", engine="python").dropna(axis=1, how="all")
            tables.append(df)
        except Exception:
            pass
    return tables

# --- Monster Parser ---
def parse_monster(md_text):
    data = {}
    # Name
    m = re.search(r"#\s*(.+)", md_text)
    if m:
        data["name"] = m.group(1).strip()

    # Kernwerte
    patterns = {
        "hg": r"HG\s*:? ([\d/.,]+)",
        "rk": r"Rüstungsklasse\s*:? (\d+)",
        "tp": r"Trefferpunkte\s*:? (\d+)",
        "initiative": r"Initiative\s*:? ([+\-]?\d+)",
        "bewegung": r"Bewegung\s*:? (.+)",
        "angriffe": r"Angriffe?\s*:? (.+)",
        "resistenz": r"Resistenz\s*:? (.+)",
        "immunitaet": r"Immunität\s*:? (.+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, md_text, flags=re.IGNORECASE)
        if m:
            data[key] = m.group(1).strip()
    return data

def insert_monster(entity_id, monster_data):
    cur.execute("""
        INSERT INTO monster (entity_id, name, hg, rk, tp, initiative, bewegung, angriffe, resistenz, immunitaet)
        VALUES (%(entity_id)s, %(name)s, %(hg)s, %(rk)s, %(tp)s, %(initiative)s, %(bewegung)s, %(angriffe)s, %(resistenz)s, %(immunitaet)s)
        RETURNING id
    """, {**monster_data, "entity_id": entity_id})
    return cur.fetchone()[0]

# --- Klassen Parser ---
def parse_klasse(md_text):
    data = {}
    m = re.search(r"#\s*(.+)", md_text)
    if m:
        data["name"] = m.group(1).strip()

    patterns = {
        "rolle": r"Rolle:\s*(.+)",
        "gesinnung": r"Gesinnung:\s*(.+)",
        "trefferwuerfel": r"Trefferwürfel.*?(W\d+)",
        "startgold": r"Startgold:\s*(.+)",
        "fertigkeiten_pro_stufe": r"Fertigkeitspunkte pro.*?:\s*(.+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, md_text, flags=re.IGNORECASE)
        if m:
            data[key] = m.group(1).strip()

    data["tables"] = extract_markdown_tables(md_text)
    return data

def insert_klasse(entity_id, klassen_data):
    cur.execute("""
        INSERT INTO klassen (entity_id, name, rolle, gesinnung, trefferwuerfel, startgold, fertigkeiten_pro_stufe, zauberfaehig)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """, (
        entity_id,
        klassen_data.get("name"),
        klassen_data.get("rolle"),
        klassen_data.get("gesinnung"),
        klassen_data.get("trefferwuerfel"),
        klassen_data.get("startgold"),
        klassen_data.get("fertigkeiten_pro_stufe"),
        True
    ))
    return cur.fetchone()[0]

def insert_klasse_tables(klasse_id, tables):
    """Versuche Tabellen den richtigen Kategorien zuzuordnen"""
    for df in tables:
        header = df.columns[1:].tolist()  # Erste Spalte oft leer
        # Tabelle mit Progression
        if "Stufe" in df.iloc[:,0].values:
            for _, row in df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO klassen_progression
                        (klasse_id, stufe, grundangriffsbonus, ref, wil, zaeh, spezial, zauber_pro_tag)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        klasse_id,
                        int(row["Stufe"]),
                        row.get("GAB"),
                        row.get("REF"),
                        row.get("WIL"),
                        row.get("ZÄH"),
                        row.get("Speziell"),
                        None  # Hier könnte man die Spalten 1–9 in JSON packen
                    ))
                except Exception:
                    pass

        # Tabelle mit bekannten Zaubern
        elif "0." in header:
            for _, row in df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO klassen_bekannte_zauber
                        (klasse_id, stufe, grad_0, grad_1, grad_2, grad_3, grad_4, grad_5, grad_6, grad_7, grad_8, grad_9)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        klasse_id,
                        int(row["Stufe"]),
                        row.get("0."),
                        row.get("1."),
                        row.get("2."),
                        row.get("3."),
                        row.get("4."),
                        row.get("5."),
                        row.get("6."),
                        row.get("7."),
                        row.get("8."),
                        row.get("9."),
                    ))
                except Exception:
                    pass

# --- Hauptlogik ---
def process_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    filename = os.path.basename(filepath).lower()
    source = "Unbekannt"

    if "monsterhandbuch" in filename:
        category = "Monster"
        data = parse_monster(text)
        entity_id = insert_entity(data.get("name"), category, source, text)
        monster_id = insert_monster(entity_id, data)

    elif "klassen" in filename:
        category = "Klasse"
        data = parse_klasse(text)
        entity_id = insert_entity(data.get("name"), category, source, text)
        klasse_id = insert_klasse(entity_id, data)
        insert_klasse_tables(klasse_id, data.get("tables", []))

    else:
        category = "Regel"
        entity_id = insert_entity(filename, category, source, text)

    # Embedding erzeugen
    embedding = create_embedding(text[:2000])  # nur ein Auszug für Performance
    insert_embedding(entity_id, embedding)

    conn.commit()

if __name__ == "__main__":
    md_dir = "prd_out/"
    for file in os.listdir(md_dir):
        if file.endswith(".md"):
            process_markdown_file(os.path.join(md_dir, file))
    print("✅ Import abgeschlossen.")
