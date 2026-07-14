#!/usr/bin/env python3
"""Build TripWise's persistent ChromaDB travel-knowledge collection."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import chromadb
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "travel_knowledge"
CSV_PATH = DATA_PATH / f"{COLLECTION_NAME}.csv"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def embed(client: OpenAI, model: str, texts: Iterable[str]) -> list[list[float]]:
    """Generate OpenAI embeddings for a collection of texts."""
    text_list = list(texts)
    if not text_list:
        return []
    response = client.embeddings.create(model=model, input=text_list)
    return [item.embedding for item in response.data]


def build_vector_db(
    model: str = OPENAI_EMBED_MODEL,
    chroma_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    csv_path: Path = CSV_PATH,
) -> None:
    """Read the travel CSV, embed its rows, and persist them in ChromaDB."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Travel knowledge file was not found: {csv_path}\n"
            "Expected columns: title, content, category."
        )

    client = OpenAI()
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    # Rebuilding avoids duplicate IDs and ensures the index matches the CSV.
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents: list[str] = []
    document_ids: list[str] = []
    metadata: list[dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"title", "content", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for index, row in enumerate(reader):
            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            category = (row.get("category") or "general").strip()
            if not title or not content:
                continue

            documents.append(f"{title}. {content}")
            document_ids.append(f"travel_doc_{index}")
            metadata.append({"source": title, "category": category})

    if not documents:
        raise ValueError(f"No usable documents were found in {csv_path}.")

    embeddings = embed(client, model, documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
        ids=document_ids,
    )

    print(
        f"Built persistent ChromaDB collection '{collection_name}' with "
        f"{len(documents)} documents at {chroma_path}"
    )


if __name__ == "__main__":
    build_vector_db()
