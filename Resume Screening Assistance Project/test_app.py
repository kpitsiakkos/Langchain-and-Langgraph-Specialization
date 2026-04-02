import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_pdf_text, create_docs, create_embeddings_load_data

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PDFS = [
    "embedded-software-engineer-resume-example.pdf",
    "python-developer-resume-example.pdf",
    "senior-programmer-resume-example.pdf",
]

def get_pdf_path(filename):
    """Resolve PDF path — adjust this folder to where your PDFs live."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)


# ── Unit Tests ────────────────────────────────────────────────────────────────

class TestGetPdfText:
    def test_returns_string(self):
        path = get_pdf_path(SAMPLE_PDFS[0])
        result = get_pdf_text(path)
        assert isinstance(result, str)

    def test_text_not_empty(self):
        path = get_pdf_path(SAMPLE_PDFS[0])
        result = get_pdf_text(path)
        assert len(result.strip()) > 0

    def test_contains_expected_keywords(self):
        path = get_pdf_path("python-developer-resume-example.pdf")
        result = get_pdf_text(path)
        # Python developer resume should mention Python
        assert "Python" in result or "python" in result.lower()


class TestCreateDocs:
    def test_returns_correct_count(self):
        paths = [get_pdf_path(p) for p in SAMPLE_PDFS]
        docs = create_docs(paths, unique_id="test-uid-123")
        assert len(docs) == len(SAMPLE_PDFS)

    def test_metadata_fields_present(self):
        paths = [get_pdf_path(SAMPLE_PDFS[0])]
        docs = create_docs(paths, unique_id="test-uid-123")
        meta = docs[0].metadata
        assert "name" in meta
        assert "id" in meta
        assert "type" in meta
        assert "size" in meta
        assert "unique_id" in meta

    def test_unique_id_stored_correctly(self):
        paths = [get_pdf_path(SAMPLE_PDFS[0])]
        uid = "abc-123-xyz"
        docs = create_docs(paths, unique_id=uid)
        assert docs[0].metadata["unique_id"] == uid

    def test_page_content_not_empty(self):
        paths = [get_pdf_path(SAMPLE_PDFS[0])]
        docs = create_docs(paths, unique_id="test-uid")
        assert len(docs[0].page_content.strip()) > 0

    def test_filename_in_metadata(self):
        paths = [get_pdf_path(SAMPLE_PDFS[0])]
        docs = create_docs(paths, unique_id="test-uid")
        assert docs[0].metadata["name"] == SAMPLE_PDFS[0]


class TestCreateEmbeddings:
    def test_embeddings_created(self):
        embeddings = create_embeddings_load_data()
        assert embeddings is not None

    def test_embeddings_encode(self):
        embeddings = create_embeddings_load_data()
        result = embeddings.embed_query("Python developer with Django experience")
        assert isinstance(result, list)
        assert len(result) == 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def test_embeddings_different_for_different_texts(self):
        embeddings = create_embeddings_load_data()
        v1 = embeddings.embed_query("Python developer")
        v2 = embeddings.embed_query("Embedded C++ firmware engineer")
        assert v1 != v2


class TestAnalyzeResumesIntegration:
    """
    Basic integration test for the full analyze_resumes() Gradio handler.
    Skipped if OPENAI_API_KEY or PINECONE_API_KEY are not set.
    """

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"),
        reason="Requires OPENAI_API_KEY and PINECONE_API_KEY env vars"
    )
    def test_full_pipeline(self):
        from app import analyze_resumes
        paths = [get_pdf_path(p) for p in SAMPLE_PDFS]
        job_desc = "Looking for a Python developer with Django, REST APIs, and AWS experience."
        result, status = analyze_resumes(job_desc, "2", "resumescreening", paths)
        assert "❌" not in result
        assert "Match Score" in result

    def test_missing_job_description(self):
        from app import analyze_resumes
        result, status = analyze_resumes("", "2", "resumescreening", ["dummy.pdf"])
        assert "❌" in result

    def test_missing_pdfs(self):
        from app import analyze_resumes
        result, status = analyze_resumes("Some job description", "2", "resumescreening", None)
        assert "❌" in result

    def test_invalid_document_count(self):
        from app import analyze_resumes
        result, status = analyze_resumes("Some job description", "abc", "resumescreening", ["dummy.pdf"])
        assert "❌" in result

    def test_missing_index_name(self):
        from app import analyze_resumes
        result, status = analyze_resumes("Some job description", "2", "", ["dummy.pdf"])
        assert "❌" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
