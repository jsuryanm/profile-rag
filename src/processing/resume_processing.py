import re 
from pathlib import Path 

from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core.schema import TextNode
from llama_index.readers.file import PDFReader
from src.config.settings import settings 
from src.config.logger import logger 

_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(summary|objective|profile|about)\b", re.I), "summary"),
    (re.compile(r"\b(experience|employment|work history|career)\b", re.I), "experience"),
    (re.compile(r"\b(education|academic|qualification)\b", re.I), "education"),
    (re.compile(r"\b(skills?|competencies|technologies|tech stack)\b", re.I), "skills"),
    (re.compile(r"\b(project|portfolio)\b", re.I), "projects"),
    (re.compile(r"\b(certification|certificate|licence|license|award)\b", re.I), "certifications"),
    (re.compile(r"\b(publication|paper|research)\b", re.I), "publications"),
    (re.compile(r"\b(volunteer|community|social)\b", re.I), "volunteer"),
]


def _detect_section(text: str) -> str:
    """Return the best matching section label for a chunk of text."""
    for pattern, label in _SECTION_PATTERNS:
        if pattern.search(text):
            return label
    return "general"

def load_pdf_resume(pdf_path: str | Path) -> list:
    """
    Reads a PDF resume and returns a list of LlamaIndex Document objects.
    Each page becomes one Document so page-level metadata is preserved.

    Args:
        pdf_path: Absolute or relative path to the uploaded PDF.

    Returns:
        List of Document objects (one per page).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")
    
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)

    if not documents:
        raise ValueError(
            f"PDFReader returned no documents from: {pdf_path}. "
            "The file may be a scanned image PDF — please use a text-based PDF."
        )
    
    logger.info(f"PDFReader loaded {len(documents)} pages from {pdf_path.name}")
    return documents

def chunk_resume(documents: list, extra_metadata: dict = None) -> list[TextNode]:

    if not documents:
        raise ValueError("No documents provided to chunk_resume.")

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Assign unique IDs and detect sections
    for i, node in enumerate(nodes):

        # Unique node ID (important when reusing Chroma collections)
        node.id_ = f"{extra_metadata.get('candidate_name','candidate')}_{i}" if extra_metadata else f"candidate_{i}"

        section = _detect_section(node.get_content())
        node.metadata["section"] = section

        if extra_metadata:
            node.metadata.update(extra_metadata)

    section_counts: dict[str, int] = {}

    for node in nodes:
        s = node.metadata["section"]
        section_counts[s] = section_counts.get(s, 0) + 1

    logger.info(
        f"Resume chunked into {len(nodes)} nodes | "
        f"sections: {section_counts} | "
        f"chunk_size={settings.chunk_size} overlap={settings.chunk_overlap}"
    )

    return nodes

def process_resume(pdf_path: str | Path, candidate_name: str = "candidate") -> list[TextNode]:
    """
    End-to-end pipeline: PDF → Documents → tagged TextNodes.

    Args:
        pdf_path:       Path to the uploaded resume PDF.
        candidate_name: Used in node metadata for filtering later.

    Returns:
        List of TextNode objects ready to be indexed into ChromaDB.
    """ 
    if not pdf_path:
        raise ValueError("pdf_path must be provided to process_resume")
    documents = load_pdf_resume(pdf_path)
    metadata = {"source":"resume",
                "candidate_name":candidate_name}
    return chunk_resume(documents,extra_metadata=metadata)