#!/usr/bin/env python3
"""
Database ingestion script for human evaluation.

Creates two separate Qdrant collections:
- gski_json_pdfs: grundschutz.json + standard PDFs
- gski_xml_pdfs: XML_Kompendium_2023.xml parsed + standard PDFs

Usage:
    python scripts/ingest_databases.py --db json  # Create gski_json_pdfs
    python scripts/ingest_databases.py --db xml   # Create gski_xml_pdfs
    python scripts/ingest_databases.py --db all   # Create both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"

# Load environment variables from notebooks/.env
try:
    from dotenv import load_dotenv
    env_path = NOTEBOOKS_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables")

if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gski_pipeline import ingest_docs


def load_pdf_chunks() -> list[dict[str, Any]]:
    """Load all chunks from parsed standard PDFs."""
    standards_dir = DATA_DIR / "data_preprocessed" / "standards"
    
    if not standards_dir.exists():
        print(f"Standards directory not found: {standards_dir}")
        print("Please run 'python scripts/parse_standards_pdf.py' first.")
        return []
    
    chunks = []
    for standard_dir in sorted(standards_dir.glob("standard_200_*")):
        for chunk_file in standard_dir.glob("*.json"):
            if chunk_file.name.startswith("_"):
                continue  # Skip index files
            
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            
            # Convert to ingestion format
            chunk_id = f"pdf_{standard_dir.name}_{chunk_data.get('id', chunk_file.stem)}"
            text = chunk_data.get("content", "")
            header = chunk_data.get("header", "")
            
            if header and text:
                text = f"{header}\n\n{text}"
            
            if text.strip():
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "meta": {
                        "source": "pdf",
                        "document": standard_dir.name,
                        "header": header,
                        "pages_start": chunk_data.get("pages", {}).get("start"),
                        "pages_end": chunk_data.get("pages", {}).get("end"),
                    }
                })
    
    print(f"Loaded {len(chunks)} chunks from PDFs")
    return chunks


def load_grundschutz_json() -> list[dict[str, Any]]:
    """Load chunks from preprocessed grundschutz.json."""
    json_path = DATA_DIR / "data_preprocessed" / "grundschutz.json"
    
    if not json_path.exists():
        print(f"grundschutz.json not found: {json_path}")
        print("Please run 'python scripts/parse_grundschutz.py' first.")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = []
    
    # Process Rollen (Roles)
    for rolle in data.get("rollen", []):
        text = f"Rolle: {rolle['name']}\n\n{rolle.get('beschreibung', '')}"
        chunks.append({
            "id": f"json_rolle_{rolle['id']}",
            "text": text,
            "meta": {
                "source": "json",
                "type": "rolle",
                "name": rolle["name"]
            }
        })
    
    # Process Elementare Gefährdungen
    for gef in data.get("elementare_gefaehrdungen", []):
        text = f"Gefährdung {gef['id']}: {gef['titel']}\n\n{gef.get('beschreibung', '')}"
        chunks.append({
            "id": f"json_gefaehrdung_{gef['id']}",
            "text": text,
            "meta": {
                "source": "json",
                "type": "gefaehrdung",
                "gefaehrdung_id": gef["id"],
                "titel": gef["titel"]
            }
        })
    
    # Process Bausteine (Modules)
    for schicht in data.get("schichten", []):
        for baustein in schicht.get("bausteine", []):
            baustein_id = baustein.get("id", "unknown")
            baustein_name = baustein.get("name", "")
            
            # Beschreibung sections
            beschreibung = baustein.get("beschreibung", {})
            for section_name, section_content in beschreibung.items():
                if section_content:
                    text = f"Baustein {baustein_id} {baustein_name}\n{section_name}\n\n{section_content}"
                    chunks.append({
                        "id": f"json_baustein_{baustein_id}_{section_name}",
                        "text": text,
                        "meta": {
                            "source": "json",
                            "type": "baustein_beschreibung",
                            "baustein_id": baustein_id,
                            "baustein_name": baustein_name,
                            "section": section_name,
                            "schicht": schicht.get("id")
                        }
                    })
            
            # Gefährdungslage
            for gef in baustein.get("gefaehrdungslage", []):
                text = f"Baustein {baustein_id} {baustein_name}\nGefährdungslage: {gef.get('titel', '')}\n\n{gef.get('beschreibung', '')}"
                chunks.append({
                    "id": f"json_baustein_{baustein_id}_gef_{gef.get('id', 'unknown')}",
                    "text": text,
                    "meta": {
                        "source": "json",
                        "type": "baustein_gefaehrdung",
                        "baustein_id": baustein_id,
                        "baustein_name": baustein_name,
                        "schicht": schicht.get("id")
                    }
                })
            
            # Anforderungen (Requirements) - stored as dict with basis/standard/erhoeht keys
            anforderungen_dict = baustein.get("anforderungen", {})
            if isinstance(anforderungen_dict, dict):
                for stufe in ["basis", "standard", "erhoeht"]:
                    for anf in anforderungen_dict.get(stufe, []):
                        if not isinstance(anf, dict):
                            continue
                        anf_id = anf.get("id", "unknown")
                        anf_info = anf.get("anforderung", {})
                        titel = anf_info.get("titel", "") if isinstance(anf_info, dict) else ""
                        typ = anf_info.get("typ_lang", stufe) if isinstance(anf_info, dict) else stufe
                        inhalt = anf.get("inhalt", "")
                        
                        text = f"Baustein {baustein_id} {baustein_name}\nAnforderung {anf_id} ({typ}): {titel}\n\n{inhalt}"
                        chunks.append({
                            "id": f"json_anforderung_{anf_id}",
                            "text": text,
                            "meta": {
                                "source": "json",
                                "type": "anforderung",
                                "anforderung_id": anf_id,
                                "titel": titel,
                                "stufe": stufe,
                                "baustein_id": baustein_id,
                                "baustein_name": baustein_name,
                                "schicht": schicht.get("id")
                            }
                        })
    
    print(f"Loaded {len(chunks)} chunks from grundschutz.json")
    return chunks


def load_xml_kompendium() -> list[dict[str, Any]]:
    """Load and parse XML_Kompendium_2023.xml into chunks."""
    xml_path = DATA_DIR / "data_raw" / "XML_Kompendium_2023.xml"
    
    if not xml_path.exists():
        print(f"XML file not found: {xml_path}")
        return []
    
    try:
        from lxml import etree
    except ImportError:
        print("lxml is required for XML parsing. Install with: pip install lxml")
        return []
    
    # Parse XML
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    # Namespace for DocBook 5.0
    NS = {"db": "http://docbook.org/ns/docbook"}
    
    chunks = []
    
    def extract_text(element) -> str:
        """Extract all text from an element."""
        if element is None:
            return ""
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.append(extract_text(child))
            if child.tail:
                texts.append(child.tail)
        return "".join(texts).strip()
    
    # Process all sections/chapters
    chunk_idx = 0
    for chapter in root.findall(".//db:chapter", NS) + root.findall(".//db:section", NS):
        title_elem = chapter.find("db:title", NS)
        title = extract_text(title_elem) if title_elem is not None else f"Section {chunk_idx}"
        
        # Get all paragraphs in this section
        content_parts = []
        for para in chapter.findall(".//db:para", NS):
            text = extract_text(para)
            if text:
                content_parts.append(text)
        
        if content_parts:
            full_text = f"{title}\n\n" + "\n\n".join(content_parts)
            
            # Split into reasonable chunks (max ~4000 chars)
            if len(full_text) > 4000:
                # Simple chunking by paragraphs
                current_chunk = title + "\n\n"
                for para in content_parts:
                    if len(current_chunk) + len(para) > 4000 and len(current_chunk) > 100:
                        chunks.append({
                            "id": f"xml_chunk_{chunk_idx}",
                            "text": current_chunk.strip(),
                            "meta": {
                                "source": "xml",
                                "title": title
                            }
                        })
                        chunk_idx += 1
                        current_chunk = title + "\n\n" + para + "\n\n"
                    else:
                        current_chunk += para + "\n\n"
                
                if len(current_chunk.strip()) > 100:
                    chunks.append({
                        "id": f"xml_chunk_{chunk_idx}",
                        "text": current_chunk.strip(),
                        "meta": {
                            "source": "xml",
                            "title": title
                        }
                    })
                    chunk_idx += 1
            else:
                chunks.append({
                    "id": f"xml_chunk_{chunk_idx}",
                    "text": full_text,
                    "meta": {
                        "source": "xml",
                        "title": title
                    }
                })
                chunk_idx += 1
    
    print(f"Loaded {len(chunks)} chunks from XML Kompendium")
    return chunks


def create_database_json_pdfs():
    """Create gski_json_pdfs collection (grundschutz.json + PDFs)."""
    print("\n" + "=" * 60)
    print("Creating Database A: gski_json_pdfs")
    print("Source: grundschutz.json + standard PDFs")
    print("=" * 60 + "\n")
    
    chunks = []
    chunks.extend(load_grundschutz_json())
    chunks.extend(load_pdf_chunks())
    
    if not chunks:
        print("ERROR: No chunks to ingest!")
        return False
    
    print(f"\nTotal chunks: {len(chunks)}")
    print("Ingesting into Qdrant collection 'gski_json_pdfs'...")
    
    count = ingest_docs(chunks, collection_name="gski_json_pdfs", recreate=True)
    print(f"Successfully ingested {count} documents")
    return True


def create_database_xml_pdfs():
    """Create gski_xml_pdfs collection (XML + PDFs)."""
    print("\n" + "=" * 60)
    print("Creating Database B: gski_xml_pdfs")
    print("Source: XML_Kompendium_2023.xml + standard PDFs")
    print("=" * 60 + "\n")
    
    chunks = []
    chunks.extend(load_xml_kompendium())
    chunks.extend(load_pdf_chunks())
    
    if not chunks:
        print("ERROR: No chunks to ingest!")
        return False
    
    print(f"\nTotal chunks: {len(chunks)}")
    print("Ingesting into Qdrant collection 'gski_xml_pdfs'...")
    
    count = ingest_docs(chunks, collection_name="gski_xml_pdfs", recreate=True)
    print(f"Successfully ingested {count} documents")
    return True


def load_baseline_chunks(chunk_size: int = 2000, chunk_overlap: int = 200) -> list[dict[str, Any]]:
    """
    Load XML Kompendium using character-based chunking (baseline approach).
    
    This mirrors the approach in 01_rag_baseline.ipynb:
    - Extract all text from XML
    - Split into fixed-size chunks with overlap
    - Minimal metadata
    """
    xml_path = DATA_DIR / "data_raw" / "XML_Kompendium_2023.xml"
    
    if not xml_path.exists():
        print(f"XML file not found: {xml_path}")
        return []
    
    try:
        from lxml import etree
    except ImportError:
        print("lxml is required for XML parsing. Install with: pip install lxml")
        return []
    
    # Parse XML with recovery mode (handles malformed XML)
    raw = xml_path.read_bytes()
    
    # Remove UTF-8 BOM if present
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    
    # Skip non-XML header if present
    lt = text.find("<")
    if lt > 0:
        text = text[lt:]
    
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    try:
        root = etree.fromstring(text.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError as exc:
        print(f"XML parsing failed: {exc}")
        return []
    
    # Extract all text nodes
    texts = [t.strip() for t in root.itertext() if t and t.strip()]
    joined = "\n".join(texts)
    
    print(f"Extracted {len(texts)} text nodes, {len(joined)} characters total")
    
    # Character-based chunking with overlap
    chunks = []
    start = 0
    chunk_idx = 0
    
    while start < len(joined):
        end = start + chunk_size
        chunk_text = joined[start:end]
        
        if chunk_text.strip():
            chunks.append({
                "id": f"baseline_chunk_{chunk_idx}",
                "text": chunk_text,
                "meta": {
                    "source": "baseline",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "char_start": start,
                    "char_end": end,
                }
            })
            chunk_idx += 1
        
        start = end - chunk_overlap
    
    print(f"Created {len(chunks)} baseline chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_database_baseline():
    """Create gski_baseline collection (character-based chunking like 01_rag_baseline.ipynb)."""
    print("\n" + "=" * 60)
    print("Creating Database C: gski_baseline")
    print("Source: XML_Kompendium_2023.xml (character-based chunking)")
    print("Chunk size: 2000, Overlap: 200")
    print("=" * 60 + "\n")
    
    chunks = load_baseline_chunks(chunk_size=2000, chunk_overlap=200)
    
    if not chunks:
        print("ERROR: No chunks to ingest!")
        return False
    
    print(f"\nTotal chunks: {len(chunks)}")
    print("Ingesting into Qdrant collection 'gski_baseline'...")
    
    count = ingest_docs(chunks, collection_name="gski_baseline", recreate=True)
    print(f"Successfully ingested {count} documents")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--db", 
        choices=["json", "xml", "baseline", "all"],
        required=True,
        help="Which database to create: 'json' (grundschutz.json+PDFs), 'xml' (XML+PDFs), 'baseline' (character chunking), or 'all'"
    )
    args = parser.parse_args()
    
    if args.db == "json":
        success = create_database_json_pdfs()
    elif args.db == "xml":
        success = create_database_xml_pdfs()
    elif args.db == "baseline":
        success = create_database_baseline()
    else:  # all
        success1 = create_database_json_pdfs()
        success2 = create_database_xml_pdfs()
        success3 = create_database_baseline()
        success = success1 and success2 and success3
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
