"""
memory_system.obsidian_bridge.exporter

Reads the memory DB and writes Obsidian-compatible Markdown notes to a vault.

Vault layout produced:
  {vault}/memories/{id}.md       — one note per MemoryFragment
  {vault}/entities/{slug}.md     — one note per Entity, with wikilinked triples
  {vault}/patterns/{id}.md       — one note per StructuralPattern
  {vault}/_index.md              — Dataview MOC (overwritten each export)
  {vault}/_README.md             — instructions (written once, never overwritten)
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import MemoryConfig, load_config
from ..schema import Database


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Turn a qualified_name like src/foo.py::bar into a safe filename slug."""
    return re.sub(r'[/\\:*?"<>|\s]+', '__', name).strip("_") or "unknown"


def _fm_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, list):
        if not v:
            return "[]"
        items = "\n".join(f"  - {x}" for x in v)
        return f"\n{items}"
    return str(v)


def _frontmatter(fields: dict) -> str:
    lines = ["---"]
    for k, v in fields.items():
        rendered = _fm_val(v)
        if rendered.startswith("\n"):
            lines.append(f"{k}:{rendered}")
        else:
            lines.append(f"{k}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body). Minimal YAML-subset parser."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip()
    return fm, body


def _write_once(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


# ── Exporter ──────────────────────────────────────────────────────────────────

class ObsidianExporter:
    MACHINE_DIRS = ("memories", "entities", "patterns")

    def __init__(
        self,
        vault_path: str | Path,
        cfg: Optional[MemoryConfig] = None,
        project_id: Optional[str] = None,
    ):
        self.vault = Path(vault_path)
        self.cfg = cfg or load_config()
        self.project_id = project_id

    def export(self) -> dict:
        db = Database(self.cfg.storage.db_path)
        db.connect()
        try:
            return self._run(db)
        finally:
            db.close()

    def _run(self, db: Database) -> dict:
        self._init_vault_skeleton()

        # Load entities and triples once — used for wikilinks in both directions
        entities_by_id = self._load_entities(db)
        triples = db.fetchall("SELECT * FROM triples")

        # fragment_id → [entity qualified_name]
        frag_to_entities: dict[str, list[str]] = {}
        # entity_id → [(predicate, other_entity_id, direction)]
        entity_edges: dict[str, list[tuple[str, str, str]]] = {}

        for t in triples:
            subj = t["subject_id"]
            obj  = t["object_id"]
            pred = t["predicate"]
            frag = t["source_fragment"]
            if frag:
                frag_to_entities.setdefault(frag, [])
                if subj in entities_by_id:
                    frag_to_entities[frag].append(entities_by_id[subj]["qualified_name"])
                if obj in entities_by_id:
                    frag_to_entities[frag].append(entities_by_id[obj]["qualified_name"])
            entity_edges.setdefault(subj, []).append((pred, obj, "out"))
            entity_edges.setdefault(obj,  []).append((pred, subj, "in"))

        n_frags    = self._export_fragments(db, frag_to_entities)
        n_entities = self._export_entities(entities_by_id, entity_edges)
        n_patterns = self._export_patterns(db)
        self._write_index()

        return {"fragments": n_frags, "entities": n_entities, "patterns": n_patterns}

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_vault_skeleton(self) -> None:
        self.vault.mkdir(parents=True, exist_ok=True)
        for d in (*self.MACHINE_DIRS, "notes", "corrections"):
            (self.vault / d).mkdir(exist_ok=True)
        _write_once(self.vault / "_README.md", _README_CONTENT)

    # ── Fragments ─────────────────────────────────────────────────────────────

    def _export_fragments(
        self,
        db: Database,
        frag_to_entities: dict[str, list[str]],
    ) -> int:
        dest = self.vault / "memories"
        written = 0

        if self.project_id:
            rows = db.fetchall(
                "SELECT * FROM memory_fragments WHERE project_id=? AND is_deprecated=0",
                (self.project_id,),
            )
        else:
            rows = db.fetchall(
                "SELECT * FROM memory_fragments WHERE is_deprecated=0"
            )

        for row in rows:
            fid = row["id"]
            tags = [
                f"memory/{row['category']}",
                f"memory/{row['epistemic_class'] if 'epistemic_class' in row.keys() else 'observed'}",
                f"project/{row['project_id']}",
            ]
            fm = {
                "id": fid,
                "project_id": row["project_id"],
                "scope": row["scope"],
                "category": row["category"],
                "confidence": float(row["confidence"]),
                "epistemic_class": row["epistemic_class"] if "epistemic_class" in row.keys() else "observed",
                "mutation_velocity": float(row["mutation_velocity"]) if "mutation_velocity" in row.keys() else 0.0,
                "graph_centrality": float(row["graph_centrality"]),
                "access_count": int(row["access_count"]),
                "is_pinned": bool(row["is_pinned"]),
                "created": row["created_at"][:10],
                "tags": tags,
            }
            linked = list(dict.fromkeys(frag_to_entities.get(fid, [])))
            wikilinks = "  ".join(f"[[entities/{_slug(n)}|{n}]]" for n in linked[:8])

            body_parts = [row["content"]]
            if wikilinks:
                body_parts.append(f"\n## Linked Entities\n{wikilinks}")

            note = _frontmatter(fm) + "\n\n" + "\n".join(body_parts) + "\n"
            (dest / f"{fid}.md").write_text(note, encoding="utf-8")
            written += 1

        return written

    # ── Entities ──────────────────────────────────────────────────────────────

    def _load_entities(self, db: Database) -> dict[str, dict]:
        if self.project_id:
            rows = db.fetchall(
                "SELECT * FROM entities WHERE project_id=?", (self.project_id,)
            )
        else:
            rows = db.fetchall("SELECT * FROM entities")
        return {r["id"]: dict(r) for r in rows}

    def _export_entities(
        self,
        entities_by_id: dict[str, dict],
        entity_edges: dict[str, list[tuple[str, str, str]]],
    ) -> int:
        dest = self.vault / "entities"
        written = 0

        for eid, e in entities_by_id.items():
            qname = e["qualified_name"]
            fm = {
                "id": eid,
                "entity_type": e["entity_type"],
                "qualified_name": qname,
                "project_id": e["project_id"],
                "last_seen": e["last_seen"][:10],
                "tags": [f"entity/{e['entity_type']}", f"project/{e['project_id']}"],
            }

            rel_lines: list[str] = []
            for pred, other_id, direction in entity_edges.get(eid, []):
                other = entities_by_id.get(other_id)
                if not other:
                    continue
                other_link = f"[[entities/{_slug(other['qualified_name'])}|{other['qualified_name']}]]"
                if direction == "out":
                    rel_lines.append(f"- `{pred}` {other_link}")
                else:
                    rel_lines.append(f"- `<-{pred}` {other_link}")

            body_parts = [f"# {qname}"]
            if rel_lines:
                body_parts.append("\n## Relationships\n" + "\n".join(rel_lines))

            note = _frontmatter(fm) + "\n\n" + "\n".join(body_parts) + "\n"
            (dest / f"{_slug(qname)}.md").write_text(note, encoding="utf-8")
            written += 1

        return written

    # ── Patterns ──────────────────────────────────────────────────────────────

    def _export_patterns(self, db: Database) -> int:
        dest = self.vault / "patterns"
        rows = db.fetchall(
            "SELECT * FROM structural_patterns ORDER BY occurrence_count DESC"
        )
        for row in rows:
            fm = {
                "id": row["id"],
                "signature_hash": row["signature_hash"],
                "arity": int(row["arity"]),
                "occurrence_count": int(row["occurrence_count"]),
                "created": row["created_at"][:10],
                "last_seen": row["last_seen"][:10],
                "tags": ["pattern"],
            }
            body_parts = ["## Abstract Form", row["abstract_form"] or "_not yet articulated_"]
            if row["failure_mode"]:
                body_parts.append(f"\n## Known Failure Mode\n{row['failure_mode']}")
            try:
                exemplars = json.loads(row["exemplars_json"])
                if exemplars:
                    ex_lines = [f"- project `{e.get('project_id', '?')}`" for e in exemplars[:5]]
                    body_parts.append("\n## Exemplars\n" + "\n".join(ex_lines))
            except (json.JSONDecodeError, TypeError):
                pass

            note = _frontmatter(fm) + "\n\n" + "\n".join(body_parts) + "\n"
            (dest / f"{row['id']}.md").write_text(note, encoding="utf-8")

        return len(rows)

    # ── Index / MOC ───────────────────────────────────────────────────────────

    def _write_index(self) -> None:
        project_filter = (
            f'WHERE project_id = "{self.project_id}" AND '
            if self.project_id
            else "WHERE "
        )
        content = _INDEX_TEMPLATE.replace("{{PROJECT_FILTER}}", project_filter)
        content = content.replace("{{GENERATED_AT}}", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        (self.vault / "_index.md").write_text(content, encoding="utf-8")


# ── Static content ────────────────────────────────────────────────────────────

_README_CONTENT = """\
# Memory System — Obsidian Bridge

## Vault Layout

| Directory | Owner | Purpose |
|-----------|-------|---------|
| `memories/` | machine | One note per memory fragment. **Do not edit directly.** |
| `entities/` | machine | One note per code entity with wikilinked relationships. |
| `patterns/` | machine | Structural patterns extracted across projects. |
| `notes/` | **you** | Write notes here — they are ingested as explicit memories. |
| `corrections/` | **you** | Write corrections here — they override conflicting facts. |

## Writing Notes

Any `.md` file you create in `notes/` will be ingested into the memory system.
Optional frontmatter:

```yaml
---
project_id: my-project   # omit to use the watcher's default project
---

Your note content here. Be specific — the system stores this verbatim.
```

## Writing Corrections

Create a `.md` file in `corrections/` with this frontmatter:

```yaml
---
original_fact: "The pipeline has 3 stages"
corrected_fact: "The pipeline has 4 stages: extract, compress, embed, index"
project_id: my-project
---
```

After the watcher processes the file, `memory_synced_at` will be added to the frontmatter.

## Dataview Queries

See `_index.md` for pre-built Dataview queries that let you explore the memory graph.
"""

_INDEX_TEMPLATE = """\
# Memory Index
> _Auto-generated {{GENERATED_AT}}. Re-run `obsidian export` to refresh._

## High-Confidence Facts
```dataview
TABLE confidence, epistemic_class, access_count AS "accesses"
FROM "memories"
{{PROJECT_FILTER}}category = "fact" AND confidence >= 0.8
SORT confidence DESC
LIMIT 20
```

## Unstable Memories (High Mutation Velocity)
```dataview
TABLE confidence, mutation_velocity AS "velocity", epistemic_class
FROM "memories"
{{PROJECT_FILTER}}mutation_velocity > 0.05
SORT mutation_velocity DESC
LIMIT 10
```

## Simulated / Unverified
```dataview
TABLE confidence, mutation_velocity AS "velocity", created
FROM "memories"
{{PROJECT_FILTER}}epistemic_class = "simulated"
SORT created DESC
```

## Pinned Memories
```dataview
LIST
FROM "memories"
WHERE is_pinned = true
```

## Preferences
```dataview
TABLE confidence, epistemic_class
FROM "memories"
{{PROJECT_FILTER}}category = "preference"
SORT confidence DESC
```

## Recent Memories
```dataview
TABLE category, confidence, epistemic_class
FROM "memories"
{{PROJECT_FILTER}}category != "preference"
SORT created DESC
LIMIT 30
```

## Structural Patterns
```dataview
TABLE arity, occurrence_count, last_seen
FROM "patterns"
SORT occurrence_count DESC
```
"""
