"""Unit tests for knowledge_base_tool without real API or index access."""

import unittest
from unittest.mock import patch

import numpy as np

import knowledge_base_tool as tool


TEST_VISITOR_ID = "11111111-1111-4111-8111-111111111111"


def make_result(score: float, index: int = 1) -> dict:
    return {
        "score": score,
        "chunk_text": f"测试知识库内容 {index}",
        "source_file": f"C:\\private\\测试资料{index}.pdf",
        "page_number": index,
        "document_id": f"secret-document-{index}",
        "chunk_id": f"secret-chunk-{index}",
        "index_path": "C:\\private\\index.npz",
    }


class KnowledgeBaseToolTests(unittest.TestCase):
    def test_empty_query_is_rejected(self):
        with patch.object(tool, "has_knowledge_base") as has_kb:
            result = tool.knowledge_base_search(TEST_VISITOR_ID, "  \n ")

        self.assertEqual(result["error_code"], "INVALID_QUERY")
        self.assertEqual(result["results"], [])
        has_kb.assert_not_called()

    def test_overlong_query_is_rejected(self):
        result = tool.knowledge_base_search(
            TEST_VISITOR_ID,
            "x" * (tool.MAX_QUERY_LENGTH + 1),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "INVALID_QUERY")

    def test_non_string_query_is_rejected(self):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, 123)  # type: ignore[arg-type]

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "INVALID_QUERY")

    @patch.object(tool, "embed_text")
    @patch.object(tool, "has_knowledge_base", return_value=False)
    def test_no_knowledge_base_does_not_embed(self, has_kb, embed):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "什么是 RAG？")

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "no_knowledge_base")
        has_kb.assert_called_once_with(TEST_VISITOR_ID)
        embed.assert_not_called()

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0, 0.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_relevant_results_call_embedding_and_search_once(self, has_kb, embed, search):
        search.return_value = [make_result(0.48, 1), make_result(0.42, 2)]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "  什么是 RAG？  ")

        self.assertEqual(result["status"], "results")
        self.assertEqual(result["query"], "什么是 RAG？")
        self.assertEqual(result["result_count"], 2)
        has_kb.assert_called_once_with(TEST_VISITOR_ID)
        embed.assert_called_once_with("什么是 RAG？")
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["top_k"], tool.KNOWLEDGE_BASE_TOP_K)

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_only_candidates_at_or_above_threshold_are_returned(
        self,
        _has_kb,
        _embed,
        search,
    ):
        search.return_value = [
            make_result(0.48, 1),
            make_result(0.41, 2),
            make_result(0.34, 3),
            make_result(0.20, 4),
        ]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "知识库问题")

        self.assertEqual(result["status"], "results")
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(
            [item["source_file"] for item in result["results"]],
            ["测试资料1.pdf", "测试资料2.pdf"],
        )

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_only_top_candidate_can_pass_threshold(self, _has_kb, _embed, search):
        search.return_value = [
            make_result(0.48, 1),
            make_result(0.30, 2),
            make_result(0.25, 3),
            make_result(0.10, 4),
        ]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "单条证据问题")

        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["source_file"], "测试资料1.pdf")

    @patch.object(tool, "search", return_value=[make_result(0.35, 1)])
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_threshold_boundary_is_inclusive(self, _has_kb, _embed, _search):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "边界值问题")

        self.assertEqual(result["status"], "results")
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["score"], 0.35)

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_filtered_results_receive_continuous_rank_and_citation(
        self,
        _has_kb,
        _embed,
        search,
    ):
        search.return_value = [
            make_result(0.48, 1),
            make_result(0.30, 2),
            make_result(0.42, 3),
            make_result(0.20, 4),
        ]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "重新编号问题")

        self.assertEqual(
            [item["rank"] for item in result["results"]],
            [1, 2],
        )
        self.assertEqual(
            [item["citation"] for item in result["results"]],
            ["来源1", "来源2"],
        )
        self.assertEqual(
            [item["source_file"] for item in result["results"]],
            ["测试资料1.pdf", "测试资料3.pdf"],
        )

    @patch.object(
        tool,
        "search",
        return_value=[
            make_result(0.349, 1),
            make_result(0.30, 2),
            make_result(0.20, 3),
            make_result(0.10, 4),
        ],
    )
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_low_score_returns_no_chunks(self, _has_kb, _embed, _search):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "无关问题")

        self.assertEqual(result["status"], "no_relevant_results")
        self.assertEqual(result["top_score"], 0.349)
        self.assertEqual(result["results"], [])

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_invalid_scores_are_safely_ignored(self, _has_kb, _embed, search):
        invalid_values = [None, "0.99", float("nan"), float("inf")]
        search.return_value = [
            {**make_result(0.50, index), "score": value}
            for index, value in enumerate(invalid_values, start=1)
        ]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "非法分数")

        self.assertEqual(result["status"], "no_relevant_results")
        self.assertEqual(result["top_score"], 0.0)
        self.assertEqual(result["results"], [])

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", side_effect=RuntimeError("secret API failure"))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_embedding_failure_is_sanitized(self, _has_kb, _embed, search):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "测试问题")

        self.assertEqual(result["error_code"], "EMBEDDING_UNAVAILABLE")
        self.assertNotIn("secret", str(result))
        search.assert_not_called()

    @patch.object(tool, "search", side_effect=RuntimeError("C:\\private\\index.npz"))
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_search_failure_is_sanitized(self, _has_kb, _embed, _search):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "测试问题")

        self.assertEqual(result["error_code"], "INDEX_UNAVAILABLE")
        self.assertNotIn("private", str(result))

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_result_excludes_internal_fields_and_orders_citations(self, _has_kb, _embed, search):
        search.return_value = [make_result(0.60, 1), make_result(0.50, 2)]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "知识库问题")

        self.assertEqual([item["rank"] for item in result["results"]], [1, 2])
        self.assertEqual(
            [item["citation"] for item in result["results"]],
            ["来源1", "来源2"],
        )
        serialized = str(result)
        for forbidden in (
            "visitor_id",
            "document_id",
            "chunk_id",
            "index_path",
            "C:\\private",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(result["results"][0]["source_file"], "测试资料1.pdf")

    @patch.object(tool, "search")
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_non_pdf_locator_is_returned_without_fake_page(self, _has_kb, _embed, search):
        item = make_result(0.60, 1)
        item.update(
            {
                "source_file": "C:\\private\\旅行计划.md",
                "source_type": "md",
                "page_number": None,
                "locator_type": "section",
                "locator_value": "交通安排",
            }
        )
        search.return_value = [item]

        result = tool.knowledge_base_search(TEST_VISITOR_ID, "交通如何安排？")
        source = result["results"][0]

        self.assertEqual(source["source_type"], "md")
        self.assertIsNone(source["page_number"])
        self.assertEqual(source["locator_type"], "section")
        self.assertEqual(source["locator_value"], "交通安排")
        self.assertNotIn("document_id", source)
        self.assertNotIn("chunk_id", source)

    @patch.object(tool, "search", return_value=[make_result(0.60, 3)])
    @patch.object(tool, "embed_text", return_value=np.array([1.0], dtype=np.float32))
    @patch.object(tool, "has_knowledge_base", return_value=True)
    def test_legacy_pdf_result_gets_page_locator(self, _has_kb, _embed, _search):
        result = tool.knowledge_base_search(TEST_VISITOR_ID, "PDF 问题")
        source = result["results"][0]

        self.assertEqual(source["source_type"], "pdf")
        self.assertEqual(source["page_number"], 3)
        self.assertEqual(source["locator_type"], "page")
        self.assertEqual(source["locator_value"], "3")

    def test_model_schema_exposes_only_query(self):
        function_schema = tool.KNOWLEDGE_BASE_SEARCH_TOOL["function"]
        properties = function_schema["parameters"]["properties"]

        self.assertEqual(function_schema["name"], "knowledge_base_search")
        self.assertEqual(set(properties), {"query"})
        self.assertNotIn("visitor_id", str(tool.KNOWLEDGE_BASE_SEARCH_TOOL))
        self.assertNotIn("top_k", str(tool.KNOWLEDGE_BASE_SEARCH_TOOL))


if __name__ == "__main__":
    unittest.main()
