# data/

This directory contains the runtime data files for Moonlight. It is not committed to git (see `.gitignore`).

## What goes here

| File | Description |
|---|---|
| `moonlight.db` | Primary SQLite database: corpus, embeddings, glossary, translation logs |
| `geonames_mv.txt` | GeoNames export for the Maldives (MV), used to populate `place_names` table |
| `embeddings_cache/` | Optional: pre-serialised numpy arrays of the embedding matrix for faster startup |

## Building `moonlight.db`

**From a [kahzaabu](https://github.com/Sofwath/kahzaabu) database (fastest):**
```bash
# Clone kahzaabu and build kahzaabu.db per its README, then:
python -m moonlight.corpus import --source /path/to/kahzaabu.db --out data/moonlight.db
python -m moonlight.corpus embed --db data/moonlight.db
```

**From scratch:**
```bash
python -m moonlight.corpus scrape --out data/moonlight.db --delay 1.5
python -m moonlight.corpus align --db data/moonlight.db
python -m moonlight.corpus index --db data/moonlight.db
python -m moonlight.corpus embed --db data/moonlight.db
```

See [docs/DATASET.md](../docs/DATASET.md) for full corpus documentation.

## GeoNames data

Download the Maldives extract from GeoNames:
```bash
curl -o data/MV.zip https://download.geonames.org/export/dump/MV.zip
unzip data/MV.zip -d data/
```

The place name import step reads `data/MV.txt` automatically:
```bash
python -m moonlight.corpus placenames --db data/moonlight.db --geonames data/MV.txt
```

## Database size

A full build (2,648 article pairs + sentence alignments + embeddings) produces a database of approximately 400–500 MB.
