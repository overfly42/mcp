CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT CHECK (category IN ('Monster', 'Klasse', 'Fertigkeit', 'Talent', 'Zauber', 'Ausrüstung', 'Regel')),
    source TEXT,
    content TEXT
);
CREATE TABLE monster (
    id SERIAL PRIMARY KEY,
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    hg NUMERIC(4,2),           -- Herausforderungsgrad (z. B. 2 oder 1/2)
    typ TEXT,
    groesse TEXT,
    gesinnung TEXT,
    rk INT,
    tp INT,
    initiative INT,
    bewegung TEXT,
    angriffe TEXT,
    schadensreduktion TEXT,
    resistenz TEXT,
    immunitaet TEXT,
    zauberresistenz TEXT
);
CREATE TABLE monster_fertigkeiten (
    id SERIAL PRIMARY KEY,
    monster_id INT REFERENCES monster(id) ON DELETE CASCADE,
    fertigkeit TEXT NOT NULL,
    wert TEXT
);
CREATE TABLE monster_besonderheiten (
    id SERIAL PRIMARY KEY,
    monster_id INT REFERENCES monster(id) ON DELETE CASCADE,
    bezeichnung TEXT,
    beschreibung TEXT
);
CREATE TABLE klassen (
    id SERIAL PRIMARY KEY,
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    rolle TEXT,
    gesinnung TEXT,
    trefferwuerfel TEXT,        -- z. B. W6
    startgold TEXT,
    fertigkeiten_pro_stufe TEXT,
    grundangriffsbonus TEXT,
    rettungswuerfe TEXT,
    zauberfaehig BOOLEAN,
    zauberquelle TEXT,
    zauberliste TEXT,
    besonderheiten TEXT
);
CREATE TABLE klassen_progression (
    id SERIAL PRIMARY KEY,
    klasse_id INT REFERENCES klassen(id) ON DELETE CASCADE,
    stufe INT,
    grundangriffsbonus TEXT,
    ref TEXT,
    wil TEXT,
    zaeh TEXT,
    spezial TEXT,
    zauber_pro_tag JSONB -- Flexibel für Spalten 1–9
);
CREATE TABLE klassen_bekannte_zauber (
    id SERIAL PRIMARY KEY,
    klasse_id INT REFERENCES klassen(id) ON DELETE CASCADE,
    stufe INT,
    grad_0 INT,
    grad_1 INT,
    grad_2 INT,
    grad_3 INT,
    grad_4 INT,
    grad_5 INT,
    grad_6 INT,
    grad_7 INT,
    grad_8 INT,
    grad_9 INT
);
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE embeddings (
    entity_id INT REFERENCES entities(id) ON DELETE CASCADE,
    embedding VECTOR(384) -- Dimension muss zu deinem SentenceTransformer passen
);
CREATE INDEX ON embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);





