"""
memory_system.obsidian_bridge — Bidirectional Obsidian vault sync.

  DB → vault/memories/*.md       (exporter, machine writes)
  DB → vault/entities/*.md       (exporter, machine writes)
  DB → vault/patterns/*.md       (exporter, machine writes)
  vault/notes/*.md     → daemon  (watcher, human writes → ingest)
  vault/corrections/*.md → daemon  (watcher, human writes → corrections table)
"""
from .exporter import ObsidianExporter
from .watcher import VaultWatcher

__all__ = ["ObsidianExporter", "VaultWatcher"]
