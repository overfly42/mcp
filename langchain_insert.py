import os
import re
import glob
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List

import psycopg2
import numpy as np
import frontmatter
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# LangChain + LangGraph
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --------------- ENV VARS ------------------
load_dotenv()
DB_DSN = os.getenv('DB_DSN', 'postgresql://myuser:mypassword@localhost:5432/mydb')
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
DATA_ROOT = Path(os.getenv('DATA_ROOT', './prd_out'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '800'))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# --------------- HELPERS -------------------

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) + 1 <= chunk_size:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            if cur:
                chunks.append(' '.join(cur))
            if len(s) > chunk_size:
                for i in range(0, len(s), chunk_size):
                    chunks.append(s[i:i+chunk_size])
                cur, cur_len = [], 0
            else:
                cur, cur_len = [s], len(s)
    if cur:
        chunks.append(' '.join(cur))
    return chunks

# --------------- SCHEMAS -------------------

class MonsterSchema(BaseModel):
    cr: str = Field(default=None)
    size: str = Field(default=None)
    type: str = Field(default=None)
    alignment: str = Field(default=None)
    ac: str = Field(default=None)
    hp: str = Field(default=None)
    speed: str = Field(default=None)
    abilities: Dict[str, Any] = Field(default_factory=dict)
    defenses: Dict[str, Any] = Field(default_factory=dict)
    actions: Dict[str, Any] = Field(default_factory=dict)

class SpellSchema(BaseModel):
    level: int = Field(default=None)
    school: str = Field(default=None)
    casting_time: str = Field(default=None)
    range: str = Field(default=None)
    components: str = Field(default=None)
    duration: str = Field(default=None)
    classes: List[str] = Field(default_factory=list)

# --------------- DB HELPER -----------------

class DB:
    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)

    def close(self):
        self.conn.close()

    def insert_entity(self, entity_type, name, source_file, content, metadata, embedding, source_hash):
        cur = self.conn.cursor()
        vec = '[' + ','.join(f"{float(x):.6f}" for x in embedding) + ']'
        cur.execute("""
            INSERT INTO public.entities (entity_type,name,source_file,content,metadata,source_file_hash,embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s::vector)
            RETURNING id
        """, (entity_type, name, source_file, content, json.dumps(metadata), source_hash, vec))
        eid = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return eid

    def insert_segment(self, entity_id, idx, text, embedding):
        cur = self.conn.cursor()
        vec = '[' + ','.join(f"{float(x):.6f}" for x in embedding) + ']'
        cur.execute("""
            INSERT INTO public.segments (entity_id,chunk_index,chunk_text,chunk_hash,embedding)
            VALUES (%s,%s,%s,%s,%s::vector)
        """, (entity_id, idx, text, sha1_text(text), vec))
        self.conn.commit()
        cur.close()

    def insert_monster(self, entity_id, data: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO public.monsters (entity_id,cr,size,type,alignment,ac,hp,speed,abilities,defenses,actions)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (entity_id) DO NOTHING
        """, (entity_id, data.get('cr'), data.get('size'), data.get('type'), data.get('alignment'),
              data.get('ac'), data.get('hp'), data.get('speed'),
              json.dumps(data.get('abilities', {})),
              json.dumps(data.get('defenses', {})),
              json.dumps(data.get('actions', {}))))
        self.conn.commit()
        cur.close()

    def insert_spell(self, entity_id, data: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO public.spells (entity_id,level,school,casting_time,range,components,duration,classes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (entity_id) DO NOTHING
        """, (entity_id, data.get('level'), data.get('school'), data.get('casting_time'), data.get('range'),
              data.get('components'), data.get('duration'), json.dumps(data.get('classes', []))))
        self.conn.commit()
        cur.close()

# --------------- LLM EXTRACTION ------------

def extract_with_ollama(text: str, entity_type: str):
    if entity_type == 'monster':
        schema = MonsterSchema
    elif entity_type == 'spell':
        schema = SpellSchema
    else:
        return {}

    parser = PydanticOutputParser(pydantic_object=schema)
    prompt = PromptTemplate(
        template="""
        Extrahiere die Felder aus folgendem {entity_type}-Text im JSON-Format:
        {format_instructions}

        Text:
        {text}
        """,
        input_variables=["text", "entity_type"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llm = Ollama(model=OLLAMA_MODEL)
    chain = prompt | llm | parser

    try:
        return chain.invoke({"text": text, "entity_type": entity_type}).dict()
    except Exception as e:
        print("[WARN] LLM-Extraktion fehlgeschlagen:", e)
        return {}

# --------------- GRAPH STATE ---------------

class IngestState(dict):
    pass

# --------------- NODES ---------------------

def node_load_file(state: IngestState) -> IngestState:
    f = state["file"]
    raw = Path(f).read_text(encoding="utf-8")
    state["raw"] = raw
    return state

def node_clean_toc(state: IngestState) -> IngestState:
    state["clean"] = state["raw"][11400:]  # TOC abschneiden
    return state

def node_detect_type(state: IngestState) -> IngestState:
    md = frontmatter.loads(state["clean"])
    title = md.get("title") or md.get("name") or Path(state["file"]).stem
    content = md.content.strip()
    etype = "rule"
    if re.search(r"CR\s+\d", content):
        etype = "monster"
    elif re.search(r"Casting Time", content, re.I):
        etype = "spell"
    elif re.search(r"Hit Die", content, re.I):
        etype = "class"
    elif re.search(r"NPC", content, re.I):
        etype = "npc"
    state.update({"title": title, "content": content, "etype": etype})
    return state

def node_embed_chunks(state: IngestState) -> IngestState:
    model = state["embed_model"]
    chunks = chunk_text(state["content"])
    embeddings = [model.encode(ch) for ch in chunks]
    avg_emb = np.mean(embeddings, axis=0).tolist()
    state.update({"chunks": chunks, "embeddings": embeddings, "avg_emb": avg_emb})
    return state

def node_insert_db(state: IngestState) -> IngestState:
    db = state["db"]
    eid = db.insert_entity(state["etype"], state["title"], Path(state["file"]).name,
                           state["content"], {}, state["avg_emb"], sha1_text(state["raw"]))
    for i, ch in enumerate(state["chunks"]):
        db.insert_segment(eid, i, ch, state["embeddings"][i])
    state["entity_id"] = eid
    return state

def node_llm_extract(state: IngestState) -> IngestState:
    extracted = extract_with_ollama(state["content"], state["etype"])
    state["extracted"] = extracted
    return state

def node_insert_structured(state: IngestState) -> IngestState:
    if not state.get("extracted"):
        return state
    db = state["db"]
    if state["etype"] == "monster":
        db.insert_monster(state["entity_id"], state["extracted"])
    elif state["etype"] == "spell":
        db.insert_spell(state["entity_id"], state["extracted"])
    return state

# --------------- BUILD GRAPH ---------------

def build_graph(use_llm=False):
    g = StateGraph(IngestState)
    g.add_node("load_file", node_load_file)
    g.add_node("clean_toc", node_clean_toc)
    g.add_node("detect_type", node_detect_type)
    g.add_node("embed_chunks", node_embed_chunks)
    g.add_node("insert_db", node_insert_db)

    if use_llm:
        g.add_node("llm_extract", node_llm_extract)
        g.add_node("insert_structured", node_insert_structured)

    # Edges
    g.set_entry_point("load_file")
    g.add_edge("load_file", "clean_toc")
    g.add_edge("clean_toc", "detect_type")
    g.add_edge("detect_type", "embed_chunks")
    g.add_edge("embed_chunks", "insert_db")

    if use_llm:
        g.add_conditional_edges(
            "insert_db",
            lambda s: "llm_extract" if s["etype"] in ["monster", "spell"] else END,
            {"llm_extract": "llm_extract", END: END}
        )
        g.add_edge("llm_extract", "insert_structured")
        g.add_edge("insert_structured", END)
    else:
        g.add_edge("insert_db", END)

    return g.compile()

# --------------- MAIN ----------------------

def ingest(use_llm=False):
    model = SentenceTransformer(MODEL_NAME)
    db = DB(DB_DSN)

    files = glob.glob(str(DATA_ROOT / "*.md"))
    graph = build_graph(use_llm)

    for f in tqdm(files, desc="Ingest"):
        state = {"file": f, "embed_model": model, "db": db}
        graph.invoke(state)

    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true", help="Nutze Ollama für strukturierte Extraktion")
    args = parser.parse_args()
    ingest(use_llm=args.use_llm)
