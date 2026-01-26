#!/usr/bin/env python3
"""
Parse the Glossar (Glossary) section from IT-Grundschutz-Kompendium XML.

Extracts glossary terms with definitions for use in RAG query expansion.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from lxml import etree

# Namespace for DocBook 5.0
NS = {"db": "http://docbook.org/ns/docbook"}


def find_glossar_chapter(root: etree._Element) -> etree._Element | None:
    """Find the Glossar chapter by title."""
    for chapter in root.findall(".//db:chapter", NS):
        title_elem = chapter.find("db:title", NS)
        if title_elem is not None and title_elem.text == "Glossar":
            return chapter
    return None


def extract_text_content(element: etree._Element) -> str:
    """Extract all text content from an element, including nested elements."""
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.append(extract_text_content(child))
        if child.tail:
            texts.append(child.tail)
    return "".join(texts).strip()


def parse_glossar(xml_path: Path) -> dict:
    """
    Parse the Glossar chapter from the IT-Grundschutz XML.
    
    Returns a dictionary with metadata and list of glossary terms.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    glossar_chapter = find_glossar_chapter(root)
    if glossar_chapter is None:
        raise ValueError("Glossar chapter not found in XML")
    
    terms = []
    current_term = None
    current_definition_parts = []
    
    # Iterate through all para elements in the glossar chapter
    for para in glossar_chapter.findall("db:para", NS):
        # Check if this para contains a bold term (emphasis role="strong")
        emphasis = para.find("db:emphasis[@role='strong']", NS)
        
        if emphasis is not None:
            # Save previous term if exists
            if current_term is not None:
                definition = " ".join(current_definition_parts).strip()
                see_also = extract_see_also(definition)
                terms.append({
                    "term": current_term,
                    "definition": definition,
                    "see_also": see_also
                })
            
            # Start new term
            current_term = emphasis.text.strip() if emphasis.text else ""
            current_definition_parts = []
            
            # Check if there's text after the emphasis in the same para
            if emphasis.tail:
                current_definition_parts.append(emphasis.tail.strip())
        else:
            # This is a definition paragraph
            if current_term is not None:
                text = extract_text_content(para)
                if text:
                    current_definition_parts.append(text)
    
    # Don't forget the last term
    if current_term is not None:
        definition = " ".join(current_definition_parts).strip()
        see_also = extract_see_also(definition)
        terms.append({
            "term": current_term,
            "definition": definition,
            "see_also": see_also
        })
    
    return {
        "meta": {
            "source": "IT-Grundschutz-Kompendium 2023",
            "source_file": xml_path.name,
            "parsed_at": datetime.now().isoformat(),
            "section": "Glossar",
            "term_count": len(terms)
        },
        "terms": terms
    }


def extract_see_also(definition: str) -> list[str]:
    """
    Extract cross-references from definition text.
    
    Looks for patterns like "Siehe [Term]" or "Siehe auch [Term]".
    """
    see_also = []
    
    # Pattern for "Siehe [Term]." or "Siehe auch [Term]."
    patterns = [
        r"Siehe auch\s+([^.]+)\.",
        r"Siehe\s+([^.]+)\."
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, definition)
        for match in matches:
            # Clean up and split if multiple terms
            terms = [t.strip() for t in match.split(",")]
            terms = [t.strip() for t in " und ".join(terms).split(" und ")]
            see_also.extend([t for t in terms if t])
    
    return list(set(see_also))  # Remove duplicates


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    xml_path = project_root / "data" / "data_raw" / "XML_Kompendium_2023.xml"
    output_path = project_root / "data" / "data_preprocessed" / "glossar.json"
    
    # Check if input file exists
    if not xml_path.exists():
        print(f"Error: Input file not found: {xml_path}")
        return 1
    
    print(f"Parsing glossary from: {xml_path}")
    
    # Parse
    glossar_data = parse_glossar(xml_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(glossar_data, f, ensure_ascii=False, indent=2)
    
    print(f"Parsed {glossar_data['meta']['term_count']} glossary terms")
    print(f"Output written to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
