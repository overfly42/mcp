-- Erweiterung (falls noch nicht installiert)
CREATE EXTENSION IF NOT EXISTS vector;

-- =========================
-- Haupttabelle: entities
-- =========================
CREATE TABLE public.entities (
    id SERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    source_file_hash TEXT NOT NULL,
    embedding VECTOR(384)
);

-- Index für Embeddings (ANN Suche nach ähnlichen Entities)
CREATE INDEX entities_embedding_idx
    ON public.entities
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- Optional: schneller Zugriff nach Typ + Name
CREATE INDEX entities_type_name_idx
    ON public.entities (entity_type, name);

-- =========================
-- Segmente (Chunks)
-- =========================
CREATE TABLE public.segments (
    id SERIAL PRIMARY KEY,
    entity_id INT NOT NULL REFERENCES public.entities(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    embedding VECTOR(384)
);

-- Index für Chunk-Suche (ANN Suche nach ähnlichen Segmenten)
CREATE INDEX segments_embedding_idx
    ON public.segments
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- Optional: schneller Zugriff pro Entity + Index
CREATE INDEX segments_entity_chunk_idx
    ON public.segments (entity_id, chunk_index);

-- =========================
-- Monster-Daten
-- =========================
CREATE TABLE public.monsters (
    entity_id INT PRIMARY KEY REFERENCES public.entities(id) ON DELETE CASCADE,
    cr TEXT,
    size TEXT,
    type TEXT,
    alignment TEXT,
    ac TEXT,
    hp TEXT,
    speed TEXT,
    abilities JSONB DEFAULT '{}',
    defenses JSONB DEFAULT '{}',
    actions JSONB DEFAULT '{}'
);

-- =========================
-- Spell-Daten
-- =========================
CREATE TABLE public.spells (
    entity_id INT PRIMARY KEY REFERENCES public.entities(id) ON DELETE CASCADE,
    level INT,
    school TEXT,
    casting_time TEXT,
    range TEXT,
    components TEXT,
    duration TEXT,
    classes JSONB DEFAULT '[]'
);

-- Index auf Spell-Level + Schule für schnelle Filterung
CREATE INDEX spells_level_school_idx
    ON public.spells (level, school);
