import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.document_loader import docs

SOURCE = "data/Vantis_Energy_Solutions_Knowledge_Base.pdf"

# Repeated page header/footer lines (PyPDF puts each on its own line)
NOISE_LINES = {
    "VANTIS ENERGY SOLUTIONS",
    "Pradrix AI Engineer — Round 2 Technical Assignment",
}
NOISE_RE = re.compile(r"^(Participant Knowledge Base.*|Page\s+\d+)$")
# "5. Pricing", "10. Reference Service Locations"
HEADING_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{2,80})$", re.MULTILINE)

# Only used for sections that are too long
fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
MAX_SECTION_CHARS = 1200


def clean(text: str) -> str:
    text = text.replace("\x7f", "•").replace("(cid:127)", "•")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line in NOISE_LINES or NOISE_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def build_full_text(pages):
    """Join all pages, remembering which page each section starts on."""
    full, offsets = "", []
    for d in pages:
        offsets.append((len(full), d.metadata.get("page", 0) + 1))
        full += clean(d.page_content) + "\n"
    return full, offsets


def page_at(pos, offsets):
    page = offsets[0][1]
    for start, pg in offsets:
        if pos >= start:
            page = pg
    return page


def split_into_sections(pages):
    full, offsets = build_full_text(pages)
    matches = list(HEADING_RE.finditer(full))  # text before section 1 (cover page) is skipped
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        sections.append({
            "no": int(m.group(1)),
            "title": m.group(2).strip(),
            "body": full[m.end():end].strip(),
            "page": page_at(m.start(), offsets),
        })
    return sections


def build_chunks(pages):
    chunks = []
    for s in split_into_sections(pages):
        header = f"Section {s['no']}: {s['title']}"
        meta = {"source": SOURCE, "section_no": s["no"],
                "section_title": s["title"], "page": s["page"]}
        body = s["body"]
        parts = [body] if len(body) <= MAX_SECTION_CHARS else fallback_splitter.split_text(body)
        for j, part in enumerate(parts):
            chunks.append(Document(
                page_content=f"{header}\n{part}",
                metadata={**meta, "part": j + 1, "parts": len(parts)},
            ))
    return chunks


chunks = build_chunks(docs)

if __name__ == "__main__":
    print("Total pages:", len(docs))
    print("Total chunks:", len(chunks))
    for c in chunks:
        m = c.metadata
        print(f"§{m['section_no']:>2} part {m['part']}/{m['parts']} p{m['page']} "
              f"{len(c.page_content):>5} chars  {m['section_title']}")