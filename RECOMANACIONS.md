# Recomanacions — enotropos

Avaluació de l'estat actual (data: 2026-06-30) i pla d'acció per reparar el sistema.

---

## Estat actual

El sistema **no funciona**: cap consulta RAG es pot completar. Tanmateix, la majoria
de components estan intactes. El problema es redueix a dues causes concretes.

### El que funciona

- **Corpus de PDFs**: `corpus_enotropos/` amb `Espanya/` i `Coneixement/` ✓
- **Dades extretes**: `data/extracted/Espanya/` — centenars de DOP/IGP en `.md`+`.json` ✓
- **Parents**: `data/parents/` (Espanya + Coneixement) ✓
- **`.env`**: claus `OPENCODE_GO_API_KEY` i `NVIDIA_API_KEY` configurades ✓
- **Codi base**: tests de schema/rag-helpers/embed passen (17/17 ràpids) ✓
- **Artefactes**: `data/tables.db`, `data/graph.pkl` existeixen ✓

### El que està trencat

| # | Problema | Evidència | Impacte |
|---|---|---|---|
| 1 | **ChromaDB 1.5.9 no pot llegir `data/chroma_db/`** | `PersistentClient` es penja >60s llistant col·leccions; amb path buit funciona OK | Cap consulta pot recuperar chunks |
| 2 | **Embedding local trencat** | `fastembed` + `intfloat/multilingual-e5-large`: falta `model.onnx_data` a la cache; warning de canvi de pooling (cal fastembed==0.5.1) | `EMBEDDING_PROVIDER=local` falla; el fallback NVIDIA→local també |
| 3 | **`requirements.txt` sense versions fixades** | `chromadb`, `fastembed`, `pymupdf` sense pin → versió 1.5.9 instal·lada | Entorns no reproduïbles |
| 4 | **Risc de mismatch de dimensions** | ChromaDB possiblement construït amb NVIDIA (2048d); ara `EMBEDDING_PROVIDER=local` (1024d) | Consultes fallarien per dimensions incompatibles |

---

## Pla d'acció

### 1. [Crític] Reconstruir ChromaDB amb NVIDIA NIM

Decisió: reconstruir ara amb **NVIDIA NIM** (la key ja està configurada, 2048d, més
ràpid). Documentar també el camí local (veure §2).

Els fitxers `.md`/`.json` de `data/extracted/` **ja existeixen** i són correctes, per
tant **no cal re-extreure els PDFs**: el pipeline sense `--force` salta l'extracció
(`winegpt/extract.py:240`) i va directe a chunk → embed → store.

Passos:
1. Editar `.env`: `EMBEDDING_PROVIDER=nvidia`
2. Eliminar `data/chroma_db/` (corrupte/incompatible amb chromadb 1.5.9)
3. Reconstruir índex GI (reusa `.md` existents):
   `python -m scripts.build_index --country Espanya --reset`
4. Reconstruir índex de coneixement:
   `python -m scripts.build_knowledge_index --reset`
5. Verificar amb `python -m scripts._check_chroma`

### 2. [Crític] Reparar embeddings locals (documentar alternativa)

Per usar `EMBEDDING_PROVIDER=local` en el futur:
1. Fixar `fastembed==0.5.1` al `requirements.txt` (evita el canvi de pooling)
2. Esborrar la cache ONNX: `C:\Users\Carles\AppData\Local\Temp\fastembed_cache\`
3. Redescarregar el model `intfloat/multilingual-e5-large` (~500MB)
4. Canviar `.env`: `EMBEDDING_PROVIDER=local`
5. **Reconstruir ChromaDB amb `--reset`** (dimensions diferents: 1024d vs 2048d)

> Avís: no es pot barrejar providers a la mateixa col·lecció. Canviar de provider
> sempre requereix `--reset`.

### 3. [Alt] Fixar versions a `requirements.txt`

```
pymupdf==1.24.x
pymupdf4llm==0.0.17
fasttext-langdetect==1.0.3
chromadb<2.0,>=1.5
openai>=1.0
streamlit>=1.30
python-dotenv
tqdm
fastembed==0.5.1  # només si s'usa local
```

### 4. [Alt] Verificació end-to-end

- `streamlit run winegpt/app.py` → fer una consulta real
- `python -m pytest tests/ -v` (sense els tests que toquen ChromaDB vella)
- `python -m scripts.eval --limit 5` per reprendre mètriques

### 5. [Mitjana] Millores de qualitat RAG (de l'avaluació prèvia)

Context Recall = 0.46 (baix). Accions:
- Augmentar `retrieve_k` a `top_k * 8` (ja fet: `winegpt/rag.py:435`)
- Ajustar pesos del reranking (actual 30% emb / 50% kw / 20% GI bonus)
- Indexar text dels pares, no només dels fills

### 6. [Baixa] Manteniment i CI/CD

- Crear `.github/workflows/ci.yml` amb `ruff check` + `mypy` + `pytest`
- `.env.example` ja conté els placeholders correctes ✓
- Documentar `corpus_path.txt` al README

---

## Mètriques d'avaluació (prèvies, pendents de reprendre)

| Mètrica | Puntuació | Barra |
|---|---|---|
| Faithfulness | 0.675 | ██████████████░░░░░░ |
| Context Precision | 0.548 | ███████████░░░░░░░░░ |
| Context Recall | 0.458 | █████████░░░░░░░░░░░ |

---

## Estat post-reparació (2026-06-30)

Les accions crítiques s'han executat amb èxit:

- ✅ `EMBEDDING_PROVIDER=nvidia` activat al `.env`
- ✅ `data/chroma_db/` eliminat i reconstruït amb chromadb 1.5.9
- ✅ Índex GI reconstruït: **8795 chunks** (Espanya, 149 DOP/IGP) — reusant `.md` existents
- ✅ Índex de coneixement reconstruït: **4118 chunks** (Coneixement, 16 PDFs)
- ✅ ChromaDB llegible sense penjar-se (12,913 chunks totals)
- ✅ Pipeline RAG verificat end-to-end: embed (NVIDIA) → retrieve (ChromaDB) → LLM (DeepSeek)
- ✅ Tests ràpids: 17/17 passen (schema + rag-helpers)
- ✅ Versions fixades a `requirements.txt`

### Problema de qualitat detectat (pendent)

Una consulta de prova ("varietats DOP Rioja") ha recuperat chunks d'altres DOP
en lloc de Rioja, tot i que Rioja té 180 chunks a ChromaDB. El problema és que
`_retrieve` (`winegpt/rag.py:438`) **no passa `gi_name` a `store_query`**, així
que ChromaDB no filtra per GI i el reranking no aconsegueix pujar els chunks
correctes al top-k. Això és el problema de **Context Recall = 0.46** ja
documentat a la prioritat #1 de l'avaluació prèvia.

**Acció recomanada**: passar `gi_name` a `store_query` quan es detecta un únic GI,
per filtrar directament a ChromaDB abans del reranking.

---

## Resum executiu

El sistema **torna a funcionar**: ChromaDB reconstruït amb NVIDIA NIM (12,913
chunks), pipeline RAG verificat end-to-end, tests passen. Queda pendent millorar
la qualitat de recuperació (Context Recall) passant el filtre `gi_name` a
ChromaDB, i optimitzar l'inserció de ChromaDB (lenta per lots de 100).
