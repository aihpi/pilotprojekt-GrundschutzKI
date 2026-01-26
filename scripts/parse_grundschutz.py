#!/usr/bin/env python3
"""
Parse IT-Grundschutz-Kompendium XML to structured JSON.

Extracts:
- Rollen (Role definitions)
- Elementare Gefährdungen (Elementary threats G 0.1 - G 0.47)
- Bausteine (Modules) with nested structure:
  - Beschreibung (Einleitung, Zielsetzung, Abgrenzung und Modellierung)
  - Gefährdungslage (Threat descriptions)
  - Anforderungen (Requirements: Basis, Standard, Erhöht)
  - Weiterführende Informationen
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from lxml import etree

# Namespace for DocBook 5.0
NS = {"db": "http://docbook.org/ns/docbook"}

# Layer definitions
SCHICHTEN = {
    "ISMS": {"name": "Informationssicherheitsmanagement", "typ": "prozess"},
    "ORP": {"name": "Organisation und Personal", "typ": "prozess"},
    "CON": {"name": "Konzepte und Vorgehensweisen", "typ": "prozess"},
    "OPS": {"name": "Betrieb", "typ": "prozess"},
    "DER": {"name": "Detektion und Reaktion", "typ": "prozess"},
    "APP": {"name": "Anwendungen", "typ": "system"},
    "SYS": {"name": "IT-Systeme", "typ": "system"},
    "IND": {"name": "Industrielle IT", "typ": "system"},
    "NET": {"name": "Netze und Kommunikation", "typ": "system"},
    "INF": {"name": "Infrastruktur", "typ": "system"},
}

# Regex patterns
BAUSTEIN_ID_PATTERN = re.compile(r"^([A-Z]{2,4})\.(\d+(?:\.\d+)*)\s+(.+)$")
ANFORDERUNG_ID_PATTERN = re.compile(
    r"^([A-Z]{2,4}\.\d+(?:\.\d+)*\.A\d+)\s+(.+?)\s*\(([BSH])\)(?:\s*\[(.+?)\])?$"
)
GEFAEHRDUNG_ID_PATTERN = re.compile(r"^G\s*0\.(\d+)\s+(.+)$")
CROSS_REF_PATTERN = re.compile(r"([A-Z]{2,4}\.\d+(?:\.\d+)*)")
MODAL_VERBEN = ["MUSS", "MÜSSEN", "DARF NUR", "DARF NICHT", "DÜRFEN NICHT", 
                "SOLLTE", "SOLLTEN", "SOLLTE NICHT", "SOLLTEN NICHT"]


def extract_text_content(element: etree._Element, include_emphasis: bool = True) -> str:
    """
    Extract all text content from an element, including nested elements.
    Handles linebreak processing instructions.
    """
    if element is None:
        return ""
    
    texts = []
    if element.text:
        texts.append(element.text)
    
    for child in element:
        # Handle processing instructions (like <?linebreak?>)
        if isinstance(child, etree._ProcessingInstruction):
            if child.target == "linebreak":
                texts.append("\n")
        elif include_emphasis or child.tag != f"{{{NS['db']}}}emphasis":
            texts.append(extract_text_content(child, include_emphasis))
        
        if child.tail:
            texts.append(child.tail)
    
    return "".join(texts).strip()


def extract_list_items(element: etree._Element) -> list[str]:
    """Extract text from itemizedlist elements."""
    items = []
    for listitem in element.findall(".//db:listitem", NS):
        for para in listitem.findall("db:para", NS):
            text = extract_text_content(para)
            if text:
                items.append(text)
    return items


def extract_section_content(section: etree._Element) -> str:
    """Extract all text content from a section, including lists."""
    parts = []
    
    for child in section:
        tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else None
        
        if tag == "para":
            text = extract_text_content(child)
            if text:
                parts.append(text)
        elif tag == "itemizedlist":
            items = extract_list_items(child)
            for item in items:
                parts.append(f"• {item}")
        elif tag == "simpara":
            text = extract_text_content(child)
            if text:
                parts.append(text)
    
    return "\n\n".join(parts)


def extract_modal_verben(text: str) -> list[str]:
    """Extract modal verbs from text."""
    found = []
    for verb in MODAL_VERBEN:
        if verb in text:
            found.append(verb)
    return list(set(found))


def extract_cross_references(text: str) -> list[str]:
    """Extract Baustein cross-references from text."""
    matches = CROSS_REF_PATTERN.findall(text)
    # Filter out duplicates and sort
    return sorted(set(matches))


def find_chapter_by_title(root: etree._Element, title: str) -> etree._Element | None:
    """Find a chapter by its title."""
    for chapter in root.findall(".//db:chapter", NS):
        title_elem = chapter.find("db:title", NS)
        if title_elem is not None and title_elem.text and title.lower() in title_elem.text.lower():
            return chapter
    return None


def parse_rollen(root: etree._Element) -> list[dict]:
    """Parse the Rollen (Roles) chapter."""
    rollen_chapter = find_chapter_by_title(root, "Rollen")
    if rollen_chapter is None:
        return []
    
    rollen = []
    current_role = None
    current_description_parts = []
    
    for para in rollen_chapter.findall("db:para", NS):
        emphasis = para.find("db:emphasis[@role='strong']", NS)
        
        if emphasis is not None:
            # Save previous role
            if current_role is not None:
                rollen.append({
                    "id": f"rolle_{current_role.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                    "name": current_role,
                    "beschreibung": " ".join(current_description_parts).strip()
                })
            
            # Start new role
            current_role = emphasis.text.strip() if emphasis.text else ""
            current_description_parts = []
            
            if emphasis.tail:
                current_description_parts.append(emphasis.tail.strip())
        else:
            if current_role is not None:
                text = extract_text_content(para)
                if text:
                    current_description_parts.append(text)
    
    # Don't forget the last role
    if current_role is not None:
        rollen.append({
            "id": f"rolle_{current_role.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
            "name": current_role,
            "beschreibung": " ".join(current_description_parts).strip()
        })
    
    return rollen


def parse_elementare_gefaehrdungen(root: etree._Element) -> list[dict]:
    """Parse the Elementare Gefährdungen chapter."""
    gefaehrdungen_chapter = find_chapter_by_title(root, "Elementare Gefährdungen")
    if gefaehrdungen_chapter is None:
        return []
    
    gefaehrdungen = []
    
    for section in gefaehrdungen_chapter.findall("db:section", NS):
        title_elem = section.find("db:title", NS)
        if title_elem is None:
            continue
        
        title_text = title_elem.text or ""
        match = GEFAEHRDUNG_ID_PATTERN.match(title_text)
        if not match:
            continue
        
        nummer = match.group(1)
        titel = match.group(2).strip()
        
        # Extract description
        beschreibung_parts = []
        beispiele = []
        
        in_beispiele = False
        for child in section:
            tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else None
            
            if tag == "para":
                text = extract_text_content(child)
                if text:
                    if "beispiel" in text.lower():
                        in_beispiele = True
                    elif not in_beispiele:
                        beschreibung_parts.append(text)
            elif tag == "itemizedlist":
                items = extract_list_items(child)
                if in_beispiele:
                    beispiele.extend(items)
                else:
                    for item in items:
                        beschreibung_parts.append(f"• {item}")
        
        gefaehrdungen.append({
            "id": f"G_0.{nummer}",
            "titel": titel,
            "xml_id": section.get("{http://www.w3.org/XML/1998/namespace}id"),
            "beschreibung": "\n\n".join(beschreibung_parts),
            "beispiele": beispiele
        })
    
    return gefaehrdungen


def parse_zustaendigkeiten(anforderungen_section: etree._Element) -> dict:
    """Parse the Zuständigkeiten table from an Anforderungen section."""
    zustaendigkeiten = {
        "grundsaetzlich": [],
        "weitere": []
    }
    
    table = anforderungen_section.find(".//db:informaltable", NS)
    if table is None:
        return zustaendigkeiten
    
    tbody = table.find(".//db:tbody", NS)
    if tbody is None:
        return zustaendigkeiten
    
    for row in tbody.findall("db:tr", NS):
        cells = row.findall("db:td", NS)
        if len(cells) >= 2:
            label = extract_text_content(cells[0]).lower()
            roles_text = extract_text_content(cells[1])
            roles = [r.strip() for r in roles_text.split(",")]
            
            if "grundsätzlich" in label:
                zustaendigkeiten["grundsaetzlich"] = roles
            elif "weitere" in label:
                zustaendigkeiten["weitere"] = roles
    
    return zustaendigkeiten


def parse_anforderung(section: etree._Element, baustein_id: str, anforderungstyp: str,
                      zustaendigkeiten: dict) -> dict | None:
    """Parse a single Anforderung section."""
    title_elem = section.find("db:title", NS)
    if title_elem is None:
        return None
    
    title_text = title_elem.text or ""
    match = ANFORDERUNG_ID_PATTERN.match(title_text)
    if not match:
        return None
    
    anforderung_id = match.group(1)
    titel = match.group(2).strip()
    typ_kurz = match.group(3)
    verantwortliche_text = match.group(4)
    
    # Parse verantwortliche if present
    verantwortliche = None
    if verantwortliche_text:
        verantwortliche = [v.strip() for v in verantwortliche_text.split(",")]
    
    # Extract content
    inhalt = extract_section_content(section)
    
    # Map type
    typ_lang_map = {
        "B": "Basis-Anforderung",
        "S": "Standard-Anforderung",
        "H": "Anforderung bei erhöhtem Schutzbedarf"
    }
    
    return {
        "id": anforderung_id,
        "xml_id": section.get("{http://www.w3.org/XML/1998/namespace}id"),
        "typ": "anforderung",
        "schicht": None,  # Will be filled by caller
        "baustein": None,  # Will be filled by caller
        "anforderung": {
            "titel": titel,
            "typ": typ_kurz,
            "typ_lang": typ_lang_map.get(typ_kurz, anforderungstyp),
            "verantwortliche": verantwortliche
        },
        "zustaendigkeiten": zustaendigkeiten,
        "inhalt": inhalt,
        "modal_verben": extract_modal_verben(inhalt),
        "cross_references": extract_cross_references(inhalt)
    }


def parse_gefaehrdungslage(gefaehrdungslage_section: etree._Element, baustein_id: str) -> list[dict]:
    """Parse the Gefährdungslage section of a Baustein."""
    gefaehrdungen = []
    counter = 1
    
    for section in gefaehrdungslage_section.findall("db:section", NS):
        title_elem = section.find("db:title", NS)
        if title_elem is None:
            continue
        
        titel = extract_text_content(title_elem)
        beschreibung = extract_section_content(section)
        
        gefaehrdungen.append({
            "id": f"{baustein_id}_GF_{counter}",
            "xml_id": section.get("{http://www.w3.org/XML/1998/namespace}id"),
            "typ": "gefaehrdung_baustein",
            "titel": titel,
            "beschreibung": beschreibung,
            "schicht": None,  # Will be filled by caller
            "baustein": None  # Will be filled by caller
        })
        counter += 1
    
    return gefaehrdungen


def parse_beschreibung(beschreibung_section: etree._Element) -> dict:
    """Parse the Beschreibung section of a Baustein."""
    beschreibung = {
        "einleitung": None,
        "zielsetzung": None,
        "abgrenzung_und_modellierung": None
    }
    
    for section in beschreibung_section.findall("db:section", NS):
        title_elem = section.find("db:title", NS)
        if title_elem is None:
            continue
        
        title = (title_elem.text or "").lower()
        content = extract_section_content(section)
        
        if "einleitung" in title:
            beschreibung["einleitung"] = content
        elif "zielsetzung" in title:
            beschreibung["zielsetzung"] = content
        elif "abgrenzung" in title or "modellierung" in title:
            beschreibung["abgrenzung_und_modellierung"] = content
    
    return beschreibung


def parse_weiterfuehrende_informationen(section: etree._Element) -> dict:
    """Parse the Weiterführende Informationen section."""
    info = {
        "wissenswertes": None,
        "weitere_abschnitte": []
    }
    
    for subsection in section.findall("db:section", NS):
        title_elem = subsection.find("db:title", NS)
        if title_elem is None:
            continue
        
        title = title_elem.text or ""
        content = extract_section_content(subsection)
        
        if "wissenswertes" in title.lower():
            info["wissenswertes"] = content
        else:
            info["weitere_abschnitte"].append({
                "titel": title,
                "inhalt": content
            })
    
    return info


def parse_baustein(baustein_section: etree._Element, schicht_info: dict) -> dict:
    """Parse a complete Baustein."""
    title_elem = baustein_section.find("db:title", NS)
    if title_elem is None:
        return None
    
    title_text = title_elem.text or ""
    match = BAUSTEIN_ID_PATTERN.match(title_text)
    if not match:
        return None
    
    schicht_id = match.group(1)
    baustein_nummer = match.group(2)
    baustein_titel = match.group(3).strip()
    baustein_id = f"{schicht_id}.{baustein_nummer}"
    
    baustein_info = {
        "id": baustein_id,
        "titel": baustein_titel,
        "xml_id": baustein_section.get("{http://www.w3.org/XML/1998/namespace}id")
    }
    
    baustein = {
        "id": baustein_id,
        "xml_id": baustein_section.get("{http://www.w3.org/XML/1998/namespace}id"),
        "titel": baustein_titel,
        "schicht": schicht_info,
        "cross_references": [],
        "beschreibung": None,
        "gefaehrdungslage": [],
        "anforderungen": {
            "zustaendigkeiten": {"grundsaetzlich": [], "weitere": []},
            "basis": [],
            "standard": [],
            "erhoeht": []
        },
        "weiterfuehrende_informationen": None
    }
    
    # Parse each main section
    for section in baustein_section.findall("db:section", NS):
        section_title_elem = section.find("db:title", NS)
        if section_title_elem is None:
            continue
        
        section_title = (section_title_elem.text or "").lower()
        
        if "beschreibung" in section_title:
            baustein["beschreibung"] = parse_beschreibung(section)
            # Extract cross-references from Abgrenzung section
            if baustein["beschreibung"]["abgrenzung_und_modellierung"]:
                baustein["cross_references"] = extract_cross_references(
                    baustein["beschreibung"]["abgrenzung_und_modellierung"]
                )
                # Remove self-reference
                baustein["cross_references"] = [
                    ref for ref in baustein["cross_references"] 
                    if ref != baustein_id
                ]
        
        elif "gefährdungslage" in section_title:
            gefaehrdungen = parse_gefaehrdungslage(section, baustein_id)
            for g in gefaehrdungen:
                g["schicht"] = schicht_info
                g["baustein"] = baustein_info
            baustein["gefaehrdungslage"] = gefaehrdungen
        
        elif "anforderungen" in section_title and "erhöht" not in section_title:
            # Parse Zuständigkeiten
            zustaendigkeiten = parse_zustaendigkeiten(section)
            baustein["anforderungen"]["zustaendigkeiten"] = zustaendigkeiten
            
            # Parse requirement subsections
            for subsection in section.findall("db:section", NS):
                subsection_title_elem = subsection.find("db:title", NS)
                if subsection_title_elem is None:
                    continue
                
                subsection_title = (subsection_title_elem.text or "").lower()
                
                if "basis" in subsection_title:
                    anforderungstyp = "basis"
                elif "standard" in subsection_title:
                    anforderungstyp = "standard"
                elif "erhöht" in subsection_title:
                    anforderungstyp = "erhoeht"
                else:
                    continue
                
                for anf_section in subsection.findall("db:section", NS):
                    anforderung = parse_anforderung(
                        anf_section, baustein_id, anforderungstyp, zustaendigkeiten
                    )
                    if anforderung:
                        anforderung["schicht"] = schicht_info
                        anforderung["baustein"] = baustein_info
                        baustein["anforderungen"][anforderungstyp].append(anforderung)
        
        elif "weiterführend" in section_title:
            baustein["weiterfuehrende_informationen"] = parse_weiterfuehrende_informationen(section)
    
    return baustein


def parse_schichten(root: etree._Element) -> list[dict]:
    """Parse all Schichten (layers) with their Bausteine."""
    schichten = []
    
    for chapter in root.findall(".//db:chapter", NS):
        title_elem = chapter.find("db:title", NS)
        if title_elem is None:
            continue
        
        title_text = title_elem.text or ""
        
        # Check if this is a Schicht chapter
        schicht_id = None
        for sid in SCHICHTEN:
            if title_text.startswith(sid + " "):
                schicht_id = sid
                break
        
        if schicht_id is None:
            continue
        
        schicht_def = SCHICHTEN[schicht_id]
        schicht_info = {
            "id": schicht_id,
            "name": schicht_def["name"],
            "typ": schicht_def["typ"]
        }
        
        schicht = {
            "id": schicht_id,
            "name": schicht_def["name"],
            "typ": schicht_def["typ"],
            "xml_id": chapter.get("{http://www.w3.org/XML/1998/namespace}id"),
            "bausteine": []
        }
        
        # Parse Bausteine in this Schicht
        for section in chapter.findall("db:section", NS):
            baustein = parse_baustein(section, schicht_info)
            if baustein:
                schicht["bausteine"].append(baustein)
        
        schichten.append(schicht)
    
    return schichten


def parse_grundschutz(xml_path: Path) -> dict:
    """
    Parse the complete IT-Grundschutz XML.
    
    Returns a dictionary with the full structure.
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    
    # Parse all components
    rollen = parse_rollen(root)
    elementare_gefaehrdungen = parse_elementare_gefaehrdungen(root)
    schichten = parse_schichten(root)
    
    # Count elements
    total_bausteine = sum(len(s["bausteine"]) for s in schichten)
    total_anforderungen = sum(
        len(b["anforderungen"]["basis"]) + 
        len(b["anforderungen"]["standard"]) + 
        len(b["anforderungen"]["erhoeht"])
        for s in schichten for b in s["bausteine"]
    )
    
    return {
        "meta": {
            "source": "IT-Grundschutz-Kompendium 2023",
            "source_file": xml_path.name,
            "parsed_at": datetime.now().isoformat(),
            "statistics": {
                "rollen": len(rollen),
                "elementare_gefaehrdungen": len(elementare_gefaehrdungen),
                "schichten": len(schichten),
                "bausteine": total_bausteine,
                "anforderungen": total_anforderungen
            }
        },
        "rollen": rollen,
        "elementare_gefaehrdungen": elementare_gefaehrdungen,
        "schichten": schichten
    }


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    xml_path = project_root / "data" / "data_raw" / "XML_Kompendium_2023.xml"
    output_path = project_root / "data" / "data_preprocessed" / "grundschutz.json"
    
    # Check if input file exists
    if not xml_path.exists():
        print(f"Error: Input file not found: {xml_path}")
        return 1
    
    print(f"Parsing IT-Grundschutz from: {xml_path}")
    
    # Parse
    data = parse_grundschutz(xml_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    stats = data["meta"]["statistics"]
    print(f"Parsed successfully:")
    print(f"  - {stats['rollen']} Rollen")
    print(f"  - {stats['elementare_gefaehrdungen']} Elementare Gefährdungen")
    print(f"  - {stats['schichten']} Schichten")
    print(f"  - {stats['bausteine']} Bausteine")
    print(f"  - {stats['anforderungen']} Anforderungen")
    print(f"Output written to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
