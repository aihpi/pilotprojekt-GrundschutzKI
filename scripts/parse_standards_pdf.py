#!/usr/bin/env python3
"""
Parse BSI Standard 200-* PDF files.

Extracts text chunks with their associated headers and subheaders,
storing each chunk as a separate JSON file.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import fitz  # pymupdf


# Header detection patterns
# Level 1: "1 Einleitung" or "1. Einleitung"
# Level 2: "1.1 Zielsetzung"
# Level 3: "1.1.1 Details"
# Level 4: "1.1.1.1 Sub-details"
HEADER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-ZÄÖÜ].+)$")

# Patterns for BSI-specific elements
ANFORDERUNG_PATTERN = re.compile(
    r"^([A-Z]{2,4}\.\d+(?:\.\d+)*\.A\d+)\s+(.+?)\s*\(([BSH])\)(?:\s*\[(.+?)\])?$"
)
GEFAEHRDUNG_PATTERN = re.compile(r"^G\s*0\.(\d+)\s+(.+)$")
BAUSTEIN_PATTERN = re.compile(r"^([A-Z]{2,4})\.(\d+(?:\.\d+)*)\s+(.+)$")

# Modal verbs for requirement classification
MODAL_VERBEN = ["MUSS", "MÜSSEN", "DARF NUR", "DARF NICHT", "DÜRFEN NICHT",
                "SOLLTE", "SOLLTEN", "SOLLTE NICHT", "SOLLTEN NICHT"]


def extract_modal_verben(text: str) -> list[str]:
    """Extract modal verbs from text."""
    found = []
    for verb in MODAL_VERBEN:
        if verb in text:
            found.append(verb)
    return list(set(found))


def get_header_level(number: str) -> int:
    """Get the header level from its number (e.g., '1.2.3' -> 3)."""
    return len(number.split("."))


def is_likely_header(text: str, font_size: float, is_bold: bool, avg_font_size: float) -> bool:
    """
    Determine if a text block is likely a header based on formatting.
    
    Headers typically:
    - Have larger font size
    - Are bold
    - Match header numbering pattern
    - Are relatively short
    """
    text = text.strip()
    
    # Check numbering pattern
    if HEADER_PATTERN.match(text):
        return True
    
    # Check BSI-specific patterns
    if ANFORDERUNG_PATTERN.match(text):
        return True
    if GEFAEHRDUNG_PATTERN.match(text):
        return True
    if BAUSTEIN_PATTERN.match(text):
        return True
    
    # Check formatting (larger font and bold, short text)
    if font_size > avg_font_size * 1.1 and is_bold and len(text) < 150:
        return True
    
    return False


def extract_text_blocks(page: fitz.Page) -> list[dict]:
    """
    Extract text blocks from a page with formatting information.
    """
    blocks = []
    
    # Get text with detailed information
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue
        
        block_text_parts = []
        font_sizes = []
        is_bold_count = 0
        total_spans = 0
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                line_text += text
                font_sizes.append(span.get("size", 10))
                total_spans += 1
                
                # Check if bold (font name often contains "Bold" or "bold")
                font_name = span.get("font", "").lower()
                if "bold" in font_name or "heavy" in font_name:
                    is_bold_count += 1
            
            block_text_parts.append(line_text)
        
        full_text = "\n".join(block_text_parts).strip()
        if not full_text:
            continue
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10
        is_bold = is_bold_count > total_spans / 2 if total_spans > 0 else False
        
        blocks.append({
            "text": full_text,
            "font_size": avg_font_size,
            "is_bold": is_bold,
            "bbox": block.get("bbox", [0, 0, 0, 0])
        })
    
    return blocks


def is_page_header_or_footer(text: str, y_pos: float, page_height: float) -> bool:
    """Check if text is likely a page header or footer."""
    # Check position (top 10% or bottom 10% of page)
    if y_pos < page_height * 0.1 or y_pos > page_height * 0.9:
        # Check for common header/footer patterns
        text_lower = text.lower().strip()
        
        # Page numbers
        if re.match(r"^\d+$", text_lower):
            return True
        
        # Document title patterns
        if "bsi-standard" in text_lower or "bundesamt" in text_lower:
            return True
        
        # Short text at edges is likely header/footer
        if len(text_lower) < 50:
            return True
    
    return False


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Parse a PDF file and extract structured chunks.
    
    Returns a list of chunks with headers and content.
    """
    doc = fitz.open(str(pdf_path))
    
    # First pass: calculate average font size across document
    all_font_sizes = []
    for page in doc:
        blocks = extract_text_blocks(page)
        for block in blocks:
            all_font_sizes.append(block["font_size"])
    
    avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 10
    
    # Second pass: extract content with structure
    current_headers = {}  # level -> header text
    chunks = []
    current_chunk_content = []
    current_chunk_start_page = 1
    
    for page_num, page in enumerate(doc, 1):
        page_height = page.rect.height
        blocks = extract_text_blocks(page)
        
        for block in blocks:
            text = block["text"]
            y_pos = block["bbox"][1]  # Top y position
            
            # Skip headers and footers
            if is_page_header_or_footer(text, y_pos, page_height):
                continue
            
            # Check if this is a header
            if is_likely_header(text, block["font_size"], block["is_bold"], avg_font_size):
                # Save current chunk if we have content
                if current_chunk_content:
                    chunk = create_chunk(
                        current_headers.copy(),
                        "\n\n".join(current_chunk_content),
                        current_chunk_start_page,
                        page_num
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_chunk_content = []
                
                # Parse header
                header_match = HEADER_PATTERN.match(text)
                if header_match:
                    number = header_match.group(1)
                    title = header_match.group(2).strip()
                    level = get_header_level(number)
                    
                    # Update headers: clear all lower levels
                    current_headers[level] = {"number": number, "title": title}
                    levels_to_remove = [l for l in current_headers if l > level]
                    for l in levels_to_remove:
                        del current_headers[l]
                    
                    current_chunk_start_page = page_num
                else:
                    # It's a header by formatting but not numbered
                    # Treat as a subsection indicator
                    if current_chunk_content:
                        current_chunk_content.append(f"\n### {text}\n")
                    else:
                        current_chunk_content.append(f"### {text}")
            else:
                # Regular content
                current_chunk_content.append(text)
    
    # Don't forget the last chunk
    if current_chunk_content:
        chunk = create_chunk(
            current_headers.copy(),
            "\n\n".join(current_chunk_content),
            current_chunk_start_page,
            len(doc)
        )
        if chunk:
            chunks.append(chunk)
    
    doc.close()
    return chunks


def create_chunk(headers: dict, content: str, start_page: int, end_page: int) -> Optional[dict]:
    """Create a chunk dictionary from headers and content."""
    content = content.strip()
    if not content or len(content) < 20:
        return None
    
    # Build header hierarchy
    sorted_levels = sorted(headers.keys())
    header_path = []
    for level in sorted_levels:
        h = headers[level]
        header_path.append({
            "level": level,
            "number": h["number"],
            "title": h["title"]
        })
    
    # Determine main header
    main_header = None
    if header_path:
        main = header_path[-1]
        main_header = f"{main['number']} {main['title']}"
    
    # Create chunk ID
    if header_path:
        chunk_id = header_path[-1]["number"].replace(".", "_")
    else:
        chunk_id = f"chunk_page_{start_page}"
    
    return {
        "id": chunk_id,
        "header": main_header,
        "header_hierarchy": header_path,
        "content": content,
        "pages": {
            "start": start_page,
            "end": end_page
        },
        "modal_verben": extract_modal_verben(content)
    }


def process_pdf_file(pdf_path: Path, output_dir: Path) -> dict:
    """
    Process a single PDF file and save chunks as individual files.
    
    Returns metadata about the processing.
    """
    # Create output subdirectory for this document
    doc_name = pdf_path.stem
    doc_output_dir = output_dir / doc_name
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Parsing: {pdf_path.name}")
    chunks = parse_pdf(pdf_path)
    print(f"  Found {len(chunks)} chunks")
    
    # Save each chunk as a separate file
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{chunk['id']}.json"
        chunk_path = doc_output_dir / chunk_filename
        
        # Add source metadata
        chunk["source"] = {
            "document": doc_name,
            "file": pdf_path.name
        }
        
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    # Create index file for this document
    index = {
        "document": doc_name,
        "source_file": pdf_path.name,
        "parsed_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": c["id"],
                "header": c["header"],
                "file": f"{c['id']}.json"
            }
            for c in chunks
        ]
    }
    
    index_path = doc_output_dir / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    return {
        "document": doc_name,
        "chunks": len(chunks),
        "output_dir": str(doc_output_dir)
    }


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_raw_dir = project_root / "data" / "data_raw"
    output_dir = project_root / "data" / "data_preprocessed" / "standards"
    
    # Find all standard_200_*.pdf files
    pdf_files = sorted(data_raw_dir.glob("standard_200_*.pdf"))
    
    if not pdf_files:
        print(f"No standard_200_*.pdf files found in {data_raw_dir}")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        result = process_pdf_file(pdf_path, output_dir)
        results.append(result)
        print(f"  Saved to: {result['output_dir']}")
    
    # Create overall index
    overall_index = {
        "source": "BSI Standards 200 Series",
        "parsed_at": datetime.now().isoformat(),
        "documents": results
    }
    
    overall_index_path = output_dir / "_index.json"
    with open(overall_index_path, "w", encoding="utf-8") as f:
        json.dump(overall_index, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Total documents: {len(results)}")
    print(f"Total chunks: {sum(r['chunks'] for r in results)}")
    print(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
