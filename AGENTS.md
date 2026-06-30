# AGENTS.md — enotropos

Instruccions per a agents de codi que treballin amb el projecte `enotropos`.

## Què és

Assistent RAG per a denominacions d'origen protegides (DOP) i indicacions geogràfiques protegides (IGP) de vins europeus. Pilot actual: Espanya.

El RAG també pot consultar un **corpus de coneixement general** (`corpus_enotropos/Coneixement/`) amb documents d'enologia, viticultura, etiquetatge i regulació.

## Estructura del corpus

El fitxer `corpus_path.txt` a l'arrel ha de contenir el camí absolut al corpus de PDFs. Estructura esperada::

```text
corpus_enotropos/
├── Espanya/
│   ├── DOP_Rioja/
│   │   └── DOP_Rioja.pdf
│   ├── IGP_Castilla/
│   │   └── IGP_Castilla.pdf
│   └── ...
└── Coneixement/          # corpus de coneixement general
    ├── enologia/
    ├── regulacio/ES/
    ├── regulacio/EU/
    ├── traditional_terms/
    └── Viticultura/
```

## Pipeline

1. **Extracció**: PDF → Markdown (`pymupdf4llm` per defecte; `pymupdf (fitz)` amb `--fast`).
2. **Idioma**: `fasttext-langdetect`.
3. **Chunking**: per títols Markdown, amb fallback a paràgrafs.
4. **Embeddings**: NVIDIA NIM `nvidia/llama-nemotron-embed-vl-1b-v2`.
5. **Vector store**: ChromaDB persistent a `data/chroma_db/`.
6. **RAG**: cerca semàntica + reranking híbrid + resposta amb DeepSeek V4 Flash.

## Comandes habituals

Executa tots els scripts com a mòduls des de l'arrel del projecte::

```powershell
# Tests
python -m pytest tests/ -v

# Linter
python -m ruff check winegpt/ scripts/ tests/

# Tipat estàtic
python -m mypy winegpt/ scripts/ tests/

# Índex de GI (Espanya)
python -m scripts.build_index --country Espanya

# Índex de GI amb extracció ràpida (fitz)
python -m scripts.build_index --country Espanya --fast

# Índex de coneixement
python -m scripts.build_knowledge_index

# Forçar reextracció i reiniciar ChromaDB
python -m scripts.build_knowledge_index --force --reset

# Reextracció ràpida del coneixement
python -m scripts.build_knowledge_index --force --fast

# Aplicació Streamlit
streamlit run winegpt/app.py
# o
.\start_enotropos.cmd
```

## Convencions

- Python >= 3.11, tipat estricte amb `mypy`.
- Longitud de línia: 100 caràcters (`ruff`).
- Importa mòduls pesats (`chromadb`, `openai`) de forma lazy quan sigui possible per no alentir tests.
- Els fitxers de dades generats (`data/`, `__pycache__/`, etc.) estan exclosos de git.
- No pengis `corpus_path.txt`, `.env` ni claus API.

## Dependències principals

- `pymupdf4llm` / `pymupdf`
- `fasttext-langdetect`
- `chromadb`
- `openai` (per a embeddings NVIDIA NIM i LLM)
- `streamlit`
- `python-dotenv`
- `tqdm`

## Notes importants

- El corpus de `Coneixement` s'emmagatzema a ChromaDB amb `country=Coneixement` i `gi_type=knowledge`.
- L'app de Streamlit permet seleccionar `Espanya`, `Coneixement` o `All` (tots dos corpus).
- Els camps de metadades dels chunks tenen una font única a `winegpt/schema.py` (`METADATA_FIELDS`, `ChunkMetadata`). `store.add_chunks` els construeix des d'aquesta llista; els productors (`chunk.process_json`, `knowledge.chunk_knowledge`) els omplen. Si afegeixes un camp, modifica `schema.py` i tots dos costats queden alineats.
- Els noms de carpeta `DOP_*`/`IGP_*` es parsegen amb `winegpt/schema.parse_folder_name`; no dupliquis aquesta lògica.
- La construcció del graf viu a `winegpt/graph_builder.py` (`scripts/build_graph.py` n'és un shim CLI); `winegpt/graph_rag.py` hi importa, no pas de `scripts`.
- L'agent (`winegpt/agent.py`) usa un loop de tool-calling d'OpenAI (sense LangGraph) i injecta els filtres de la UI (`country`/`gi_type`/`top_k`) a `search_vector_db`.
- Els imports pesats (`chromadb`, `openai`, `networkx`) són lazy; no els importis al top-level dels mòduls lleugers.
