"""Unit tests for visitor-isolated knowledge-base document management."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from filelock import FileLock as RealFileLock

import vector_store


VISITOR_A = "11111111-1111-4111-8111-111111111111"
VISITOR_B = "22222222-2222-4222-8222-222222222222"


def document_id(marker: str) -> str:
    return hashlib.sha256(marker.encode("utf-8")).hexdigest()


def make_chunks(
    marker: str,
    source_file: str,
    source_type: str,
    count: int,
) -> list[dict]:
    current_document_id = document_id(marker)
    chunks = []
    for index in range(count):
        is_pdf = source_type == "pdf"
        locator_type = "page" if is_pdf else "chunk"
        locator_value = str(index + 1)
        chunks.append(
            {
                "chunk_text": f"{marker} knowledge chunk {index}",
                "document_id": current_document_id,
                "chunk_id": hashlib.sha256(
                    f"{marker}:{index}".encode("utf-8")
                ).hexdigest(),
                "source_file": source_file,
                "source_type": source_type,
                "page_number": index + 1 if is_pdf else None,
                "chunk_index": index,
                "locator_type": locator_type,
                "locator_value": locator_value,
            }
        )
    return chunks


class KnowledgeBaseManagementTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(
            vector_store,
            "KNOWLEDGE_BASE_ROOT",
            Path(self.temporary_directory.name),
        )
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temporary_directory.cleanup()

    def add_document(
        self,
        visitor_id: str,
        marker: str,
        source_file: str,
        source_type: str,
        count: int = 1,
        vector: tuple[float, float] = (1.0, 0.0),
    ) -> list[dict]:
        chunks = make_chunks(marker, source_file, source_type, count)
        embeddings = np.asarray([vector] * count, dtype=np.float32)
        vector_store.add_chunks(visitor_id, chunks, embeddings)
        return chunks

    def write_legacy_pdf_index(
        self,
        visitor_id: str,
        marker: str = "legacy-pdf",
        count: int = 2,
    ) -> str:
        current_document_id = document_id(marker)
        index_path = vector_store._index_path(visitor_id, create_directory=True)
        np.savez_compressed(
            index_path,
            embeddings=np.asarray([[1.0, 0.0]] * count, dtype=np.float32),
            chunk_texts=np.asarray(
                [f"legacy PDF chunk {index}" for index in range(count)],
                dtype=np.str_,
            ),
            document_ids=np.asarray([current_document_id] * count, dtype=np.str_),
            chunk_ids=np.asarray(
                [document_id(f"legacy-chunk-{index}") for index in range(count)],
                dtype=np.str_,
            ),
            source_files=np.asarray(["legacy.pdf"] * count, dtype=np.str_),
            page_numbers=np.arange(1, count + 1, dtype=np.int32),
            chunk_indexes=np.arange(count, dtype=np.int32),
        )
        return current_document_id

    def test_empty_knowledge_base_returns_empty_list(self):
        self.assertEqual(vector_store.list_documents(VISITOR_A), [])

    def test_single_pdf_is_aggregated_once_with_correct_chunk_count(self):
        self.add_document(VISITOR_A, "pdf", "guide.pdf", "pdf", count=3)

        documents = vector_store.list_documents(VISITOR_A)

        self.assertEqual(
            documents,
            [
                {
                    "document_id": document_id("pdf"),
                    "source_file": "guide.pdf",
                    "source_type": "pdf",
                    "chunk_count": 3,
                }
            ],
        )

    def test_single_docx_is_listed_as_docx(self):
        self.add_document(VISITOR_A, "docx", "guide.docx", "docx", count=2)

        document = vector_store.list_documents(VISITOR_A)[0]

        self.assertEqual(document["source_type"], "docx")
        self.assertEqual(document["chunk_count"], 2)

    def test_all_supported_types_can_coexist(self):
        fixtures = (
            ("pdf", "guide.pdf", "pdf", 1),
            ("docx", "guide.docx", "docx", 2),
            ("txt", "notes.txt", "txt", 3),
            ("md", "plan.md", "md", 4),
        )
        for marker, filename, source_type, count in fixtures:
            self.add_document(
                VISITOR_A,
                marker,
                filename,
                source_type,
                count=count,
            )

        documents = vector_store.list_documents(VISITOR_A)

        self.assertEqual(len(documents), 4)
        self.assertEqual(
            {
                document["source_type"]: document["chunk_count"]
                for document in documents
            },
            {"pdf": 1, "docx": 2, "txt": 3, "md": 4},
        )
        for document in documents:
            self.assertEqual(
                set(document),
                {"document_id", "source_file", "source_type", "chunk_count"},
            )

    def test_list_hides_a_stored_directory_path(self):
        self.add_document(
            VISITOR_A,
            "private-path",
            "C:\\private\\guide.pdf",
            "pdf",
        )

        document = vector_store.list_documents(VISITOR_A)[0]

        self.assertEqual(document["source_file"], "guide.pdf")
        self.assertNotIn("private", str(document))

    def test_delete_one_document_keeps_other_documents_and_search_data(self):
        self.add_document(
            VISITOR_A,
            "delete-me",
            "delete.pdf",
            "pdf",
            vector=(1.0, 0.0),
        )
        self.add_document(
            VISITOR_A,
            "keep-me",
            "keep.docx",
            "docx",
            vector=(0.0, 1.0),
        )

        deleted = vector_store.delete_document(
            VISITOR_A,
            document_id("delete-me"),
        )

        self.assertTrue(deleted)
        self.assertFalse(
            vector_store.has_document(VISITOR_A, document_id("delete-me"))
        )
        self.assertTrue(vector_store.has_document(VISITOR_A, document_id("keep-me")))
        self.assertEqual(
            [item["source_file"] for item in vector_store.list_documents(VISITOR_A)],
            ["keep.docx"],
        )
        results = vector_store.search(
            VISITOR_A,
            np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=4,
        )
        self.assertTrue(results)
        self.assertTrue(all(item["document_id"] != document_id("delete-me") for item in results))

    def test_delete_last_document_removes_index(self):
        self.add_document(VISITOR_A, "only", "only.txt", "txt")

        self.assertTrue(
            vector_store.delete_document(VISITOR_A, document_id("only"))
        )

        self.assertFalse(vector_store.has_knowledge_base(VISITOR_A))
        self.assertEqual(vector_store.list_documents(VISITOR_A), [])

    def test_delete_missing_document_returns_false(self):
        self.add_document(VISITOR_A, "existing", "existing.md", "md")

        deleted = vector_store.delete_document(
            VISITOR_A,
            document_id("missing"),
        )

        self.assertFalse(deleted)
        self.assertTrue(vector_store.has_document(VISITOR_A, document_id("existing")))

    def test_invalid_visitor_and_document_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            vector_store.list_documents("../../other")
        with self.assertRaises(ValueError):
            vector_store.delete_document("../../other", document_id("valid"))
        with self.assertRaises(ValueError):
            vector_store.delete_document(VISITOR_A, "../../index.npz")

    def test_legacy_pdf_index_can_be_listed_and_deleted(self):
        legacy_document_id = self.write_legacy_pdf_index(VISITOR_A)

        documents = vector_store.list_documents(VISITOR_A)

        self.assertEqual(
            documents,
            [
                {
                    "document_id": legacy_document_id,
                    "source_file": "legacy.pdf",
                    "source_type": "pdf",
                    "chunk_count": 2,
                }
            ],
        )
        self.assertTrue(
            vector_store.delete_document(VISITOR_A, legacy_document_id)
        )
        self.assertFalse(vector_store.has_knowledge_base(VISITOR_A))

    def test_legacy_pdf_index_can_delete_one_document_and_keep_another(self):
        first_document_id = document_id("legacy-first")
        second_document_id = document_id("legacy-second")
        index_path = vector_store._index_path(VISITOR_A, create_directory=True)
        np.savez_compressed(
            index_path,
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            chunk_texts=np.asarray(["first", "second"], dtype=np.str_),
            document_ids=np.asarray(
                [first_document_id, second_document_id],
                dtype=np.str_,
            ),
            chunk_ids=np.asarray(
                [document_id("legacy-first-chunk"), document_id("legacy-second-chunk")],
                dtype=np.str_,
            ),
            source_files=np.asarray(["first.pdf", "second.pdf"], dtype=np.str_),
            page_numbers=np.asarray([1, 2], dtype=np.int32),
            chunk_indexes=np.asarray([0, 0], dtype=np.int32),
        )

        self.assertTrue(
            vector_store.delete_document(VISITOR_A, first_document_id)
        )

        self.assertTrue(vector_store.has_knowledge_base(VISITOR_A))
        self.assertEqual(
            vector_store.list_documents(VISITOR_A),
            [
                {
                    "document_id": second_document_id,
                    "source_file": "second.pdf",
                    "source_type": "pdf",
                    "chunk_count": 1,
                }
            ],
        )
        loaded = vector_store._load_index(index_path)
        self.assertEqual(
            {name: len(loaded[name]) for name in vector_store.INDEX_ARRAY_FIELDS},
            {name: 1 for name in vector_store.INDEX_ARRAY_FIELDS},
        )

    def test_deleted_document_can_be_added_again(self):
        chunks = self.add_document(VISITOR_A, "again", "again.docx", "docx")
        current_document_id = document_id("again")
        self.assertTrue(vector_store.delete_document(VISITOR_A, current_document_id))

        vector_store.add_chunks(
            VISITOR_A,
            chunks,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        self.assertTrue(vector_store.has_document(VISITOR_A, current_document_id))
        self.assertEqual(vector_store.list_documents(VISITOR_A)[0]["chunk_count"], 1)

    def test_all_metadata_arrays_keep_matching_lengths_after_delete(self):
        self.add_document(VISITOR_A, "first", "first.pdf", "pdf", count=2)
        self.add_document(VISITOR_A, "second", "second.md", "md", count=3)

        vector_store.delete_document(VISITOR_A, document_id("first"))
        index_path = vector_store._index_path(VISITOR_A)
        data = vector_store._load_index(index_path)

        lengths = {name: len(data[name]) for name in vector_store.INDEX_ARRAY_FIELDS}
        self.assertEqual(set(lengths.values()), {3})

    def test_delete_uses_the_visitor_index_lock(self):
        self.add_document(VISITOR_A, "locked", "locked.txt", "txt")
        lock_paths: list[str] = []

        def tracking_lock(path: str):
            lock_paths.append(str(path))
            return RealFileLock(path)

        with patch.object(vector_store, "FileLock", side_effect=tracking_lock):
            deleted = vector_store.delete_document(
                VISITOR_A,
                document_id("locked"),
            )

        self.assertTrue(deleted)
        self.assertEqual(len(lock_paths), 1)
        self.assertTrue(lock_paths[0].endswith("index.npz.lock"))

    def test_atomic_save_failure_does_not_damage_existing_index(self):
        self.add_document(VISITOR_A, "first", "first.pdf", "pdf")
        self.add_document(VISITOR_A, "second", "second.docx", "docx")
        before = vector_store.list_documents(VISITOR_A)

        with patch.object(
            vector_store,
            "_atomic_save",
            side_effect=vector_store.VectorStoreError("模拟写入失败"),
        ):
            with self.assertRaises(vector_store.VectorStoreError):
                vector_store.delete_document(VISITOR_A, document_id("first"))

        self.assertEqual(vector_store.list_documents(VISITOR_A), before)
        self.assertTrue(vector_store.has_document(VISITOR_A, document_id("first")))
        self.assertTrue(vector_store.has_document(VISITOR_A, document_id("second")))

    def test_visitor_cannot_delete_another_visitors_document(self):
        self.add_document(VISITOR_A, "visitor-a", "a.pdf", "pdf")
        self.add_document(VISITOR_B, "visitor-b", "b.pdf", "pdf")

        deleted = vector_store.delete_document(
            VISITOR_A,
            document_id("visitor-b"),
        )

        self.assertFalse(deleted)
        self.assertTrue(
            vector_store.has_document(VISITOR_B, document_id("visitor-b"))
        )
        self.assertEqual(
            vector_store.list_documents(VISITOR_B)[0]["source_file"],
            "b.pdf",
        )


if __name__ == "__main__":
    unittest.main()
