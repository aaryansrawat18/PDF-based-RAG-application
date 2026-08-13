import unittest

from qdrant_client.models import MatchValue, Range

from app.core.metadata import chunk_matches_filters, filters_to_qdrant


class MetadataFilterTests(unittest.TestCase):
    def test_empty_filters_are_none(self):
        self.assertIsNone(filters_to_qdrant(None))
        self.assertIsNone(filters_to_qdrant({}))
        self.assertIsNone(filters_to_qdrant({"section": None, "page_gte": None}))

    def test_section_and_page_gte(self):
        qfilter = filters_to_qdrant({"section": "Retrieval", "page_gte": 10})
        self.assertIsNotNone(qfilter)
        self.assertEqual(len(qfilter.must), 2)

        section, page = qfilter.must
        self.assertEqual(section.key, "section")
        self.assertEqual(section.match, MatchValue(value="Retrieval"))
        self.assertEqual(page.key, "page")
        self.assertEqual(page.range, Range(gte=10, lte=None))

    def test_page_range_and_document(self):
        qfilter = filters_to_qdrant(
            {"document": "rag.pdf", "page_gte": 2, "page_lte": 5}
        )
        keys = [condition.key for condition in qfilter.must]
        self.assertEqual(keys, ["document", "page"])
        page = qfilter.must[1]
        self.assertEqual(page.range, Range(gte=2, lte=5))

    def test_exact_page(self):
        qfilter = filters_to_qdrant({"page": 3})
        self.assertEqual(len(qfilter.must), 1)
        self.assertEqual(qfilter.must[0].match, MatchValue(value=3))

    def test_pydantic_model_dump(self):
        class FakeFilters:
            def model_dump(self, exclude_none=True):
                return {"section": "Introduction", "page": None}

        qfilter = filters_to_qdrant(FakeFilters())
        self.assertEqual(len(qfilter.must), 1)
        self.assertEqual(qfilter.must[0].key, "section")


class ChunkMatchesFiltersTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "section": "Retrieval",
            "document": "rag.pdf",
            "page": 12,
            "chunk_id": "chunk_12",
            "content_type": "text",
        }

    def test_no_filters_match(self):
        self.assertTrue(chunk_matches_filters(self.chunk, None))
        self.assertTrue(chunk_matches_filters(self.chunk, {}))

    def test_section_and_page_gte(self):
        self.assertTrue(
            chunk_matches_filters(
                self.chunk, {"section": "Retrieval", "page_gte": 10}
            )
        )
        self.assertFalse(
            chunk_matches_filters(
                self.chunk, {"section": "Introduction", "page_gte": 10}
            )
        )
        self.assertFalse(
            chunk_matches_filters(self.chunk, {"section": "Retrieval", "page_gte": 20})
        )


if __name__ == "__main__":
    unittest.main()
