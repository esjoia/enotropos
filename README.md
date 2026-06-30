# enotropos

Assistent de RAG (Retrieval-Augmented Generation) per a denominacions d'origen protegides (DOP) i indicacions geogràfiques protegides (IGP) de vins europeus.

Actualment implementat com a pilot per a vins espanyols, amb dades extretes de les especificacions oficials de producte publicades per la UE.

## Arquitectura

El projecte està organitzat en mòduls independents que formen un pipeline clar:

1. **Extracció** (`winegpt/extract.py): converteix PDFs a Markdown amb `pymupdf` (fitz).
2. **Detecció d'idioma** (`winegpt/language.py): detecta l'idioma de cada document.
3. **Chunking** (`winegpt/chunk.py): divideix els documents en fragments per seccions o paràgrafs.
4. **Embeddings** (`winegpt/embed.py): genera vectors amb `nvidia/llama-nemotron-embed-vl-1b-v2` via NVIDIA NIM.
5. **Vector store** (`winegpt/store.py): emmagatzema i consulta els vectors amb ChromaDB.
6. **RAG** (`winegpt/rag.py): cerca semàntica, filtratge per nom de GI, *reranking* híbrid i generació de respostes amb DeepSeek V4 Flash (OpenCode Go).
7. **App** (`winegpt/app.py): interfície de chat amb Streamlit.

## Requisits

- Python >= 3.11
- Un corpus de PDFs organitzat per carpetes `DOP_<nom>` i `IGP_<nom>` dins d'un directori per país (p. ex. `Espanya/`).
- Claus d'API per a:
  - **OpenCode Go** (LLM)
  - **NVIDIA NIM** (embeddings)

## Instal·lació

```bash
# Opció 1: amb requirements.txt
pip install -r requirements.txt

# Opció 2: com a paquet editable amb dependències de desenvolupament
pip install -e ".[dev]"
```

## Coneixement general (knowledge corpus)

A més de les especificacions de producte per GI, el projecte accepta un corpus de coneixement general sobre viticultura, enologia, etiquetatge i regulació vinícola. Aquest corpus ha de residir a::

```text
corpus_enotropos/
└── Coneixement/
    ├── enologia/
    ├── regulacio/
    │   ├── ES/
    │   └── EU/
    ├── traditional_terms/
    └── Viticultura/
```

Els PDFs són extrets a `data/extracted/Coneixement/`, fragmentats, embeddats i emmagatzemats a ChromaDB amb `gi_type=knowledge` i `country=Coneixement`, de manera que el RAG els pot recuperar juntament amb els documents de GI.

## Configuració

### 1. Variables d'entorn

Copia `.env.example` a `.env` i omple les claus:

```bash
cp .env.example .env
```

Edita `.env`:

```env
OPENCODE_GO_API_KEY=sk-...
NVIDIA_API_KEY=nvapi-...
```

### 2. Ruta al corpus

Crea `corpus_path.txt` a l'arrel del projecte amb el camí absolut al directori del corpus:

```text
C:\Ruta\Al\corpus_enotropos
```

L'estructura esperada és:

```text
corpus_enotropos/
└── Espanya/
    ├── DOP_Rioja/
    │   └── DOP_Rioja.pdf
    ├── DOP_Priorat/
    │   └── DOP_Priorat.pdf
    └── IGP_Castilla/
        └── IGP_Castilla.pdf
```

## Ús

### Construir l'índex

```bash
# Extrau els PDFs, detecta idiomes, divideix en chunks, genera embeddings i emmagatzema a ChromaDB
python -m scripts.build_index --country Espanya

# Força la reextracció i reinicia la col·lecció de ChromaDB
python -m scripts.build_index --country Espanya --force --reset

# Només mostra què faria
python -m scripts.build_index --country Espanya --dry-run
```

### Construir l'índex de coneixement

```bash
# Extrau tots els PDFs de corpus_enotropos/Coneixement/ i els afegeix a ChromaDB
python -m scripts.build_knowledge_index

# Força la reextracció i reinicia la col·lecció sencera
python -m scripts.build_knowledge_index --force --reset

# Només mostra què faria
python -m scripts.build_knowledge_index --dry-run
```

### Executar l'aplicació

```bash
# Opció 1: amb Streamlit directament
streamlit run winegpt/app.py

# Opció 2: amb l'script de Windows
start_enotropos.cmd
```

Obre http://localhost:8501 al navegador.

### Executar l'avaluació

```bash
python -m scripts.eval --limit 5
```

Això avalua el RAG amb mètriques de *faithfulness*, *answer relevancy* i *context relevancy* utilitzant un jutge LLM.

## Tests i qualitat de codi

```bash
# Tests
pytest tests/ -v

# Linter
ruff check winegpt/ scripts/ tests/

# Tipat estàtic
mypy winegpt/ scripts/ tests/
```

## Estructura del projecte

```text
enotropos/
├── winegpt/              # Codi font principal
│   ├── app.py            # Aplicació Streamlit
│   ├── chunk.py          # Divisió en fragments
│   ├── config.py         # Configuració i variables d'entorn
│   ├── embed.py          # Client d'embeddings NVIDIA NIM
│   ├── extract.py        # Extracció de PDFs
│   ├── knowledge.py      # Corpus de coneixement general
│   ├── language.py       # Detecció d'idioma
│   ├── llm.py            # Client LLM compartit
│   ├── rag.py            # Pipeline RAG
│   └── store.py          # Client ChromaDB
├── scripts/              # Scripts d'administració
│   ├── build_index.py    # Construcció de l'índex de GI
│   ├── build_knowledge_index.py  # Construcció de l'índex de coneixement
│   └── eval.py           # Avaluació del RAG
├── tests/                # Tests unitaris
├── data/                 # Sortida generada (exclosa de git)
│   ├── chroma_db/        # Base de dades vectorial
│   ├── extracted/        # Markdowns extrets
│   └── eval_questions.json
├── .env.example          # Plantilla de variables d'entorn
├── corpus_path.txt       # Camí al corpus (exclòs de git)
├── Makefile              # Comandes ràpides (orientat a Unix)
├── pyproject.toml        # Configuració del projecte
├── requirements.txt      # Dependències
└── README.md             # Aquest fitxer
```

## Notes

- El fitxer `.env` i `corpus_path.txt` estan exclosos de git per `.gitignore`. No hi pengis claus ni rutes locals.
- Les dades generades (`data/`, `__pycache__/`, etc.) també estan excloses de git.
- El `Makefile` està orientat a entorns Unix; a Windows pots usar els scripts `*.cmd` o les comandes `python` directament.
