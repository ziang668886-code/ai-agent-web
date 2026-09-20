"""Unit tests for agent_service without real Doubao or Embedding calls."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent_service as service
import knowledge_base_tool as kb_tool


TEST_VISITOR_ID = "11111111-1111-4111-8111-111111111111"
OTHER_VISITOR_ID = "22222222-2222-4222-8222-222222222222"


def response_with_content(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(
    arguments: str,
    name: str = service.TOOL_NAME,
    call_id: str = "call-1",
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def response_with_tool_calls(*calls):
    message = SimpleNamespace(content=None, tool_calls=list(calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(*responses):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(side_effect=list(responses)),
            )
        )
    )
    return client


def result_status(status: str) -> dict:
    if status == "results":
        return {
            "ok": True,
            "status": "results",
            "results": [
                {
                    "citation": "来源1",
                    "rank": 1,
                    "content": "RAG 是检索增强生成。",
                    "source_file": "知识库.pdf",
                    "page_number": 2,
                    "score": 0.48,
                },
                {
                    "citation": "来源2",
                    "rank": 2,
                    "content": "补充内容。",
                    "source_file": "知识库.pdf",
                    "page_number": 2,
                    "score": 0.44,
                },
            ],
        }
    if status == "no_knowledge_base":
        return {"ok": True, "status": status, "results": []}
    if status == "no_relevant_results":
        return {"ok": True, "status": status, "top_score": 0.2, "results": []}
    return {
        "ok": False,
        "status": "temporarily_unavailable",
        "error_code": "INDEX_UNAVAILABLE",
        "message": "知识库检索暂时不可用。",
        "results": [],
    }


def weather_status(status: str) -> dict:
    if status == "success":
        return {
            "ok": True,
            "status": "success",
            "location": "郑州",
            "weather": "晴",
            "temperature_c": 28,
            "feels_like_c": 29,
            "humidity": 45,
            "wind": "东北风 2级",
            "observation_time": None,
        }
    if status == "invalid_request":
        return {
            "ok": False,
            "status": "invalid_request",
            "error_code": "INVALID_LOCATION",
            "message": "请输入有效的城市或地区名称。",
        }
    return {
        "ok": False,
        "status": "temporarily_unavailable",
        "error_code": "WEATHER_API_UNAVAILABLE",
        "message": "天气服务暂时不可用。",
    }


def poi_status(status: str = "results") -> dict:
    if status == "results":
        return {
            "ok": True,
            "status": "results",
            "query": "餐厅",
            "city": "北京",
            "search_mode": "nearby",
            "anchor": "故宫",
            "result_count": 1,
            "results": [
                {
                    "rank": 1,
                    "name": "景运门故宫餐厅",
                    "address": "故宫博物院内",
                    "district": "东城区",
                    "category": "餐饮服务;中餐厅",
                    "distance_m": 142,
                    "rating": 3.9,
                    "cost_per_person": 63,
                    "tags": ["中餐"],
                    "opening_hours": "08:30-15:30",
                    "location": {"longitude": 116.398455, "latitude": 39.918509},
                }
            ],
        }
    return {
        "ok": False,
        "status": status,
        "error_code": "POI_API_UNAVAILABLE",
        "message": "地点搜索服务暂时不可用。",
        "results": [],
    }


class AgentServiceTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "原有 System Prompt"},
            {"role": "user", "content": "什么是 RAG？"},
        ]

    @patch.object(service, "knowledge_base_search")
    def test_direct_answer_calls_llm_once_and_no_tool(self, search_tool):
        client = make_client(response_with_content("普通回答"))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "红烧肉怎么做？",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["sources"], [])
        self.assertEqual(client.chat.completions.create.call_count, 1)
        first_messages = client.chat.completions.create.call_args.kwargs["messages"]
        current_question_count = sum(
            message.get("role") == "user"
            and message.get("content") == "红烧肉怎么做？"
            for message in first_messages
        )
        self.assertEqual(current_question_count, 1)
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["tool_choice"],
            "auto",
        )
        search_tool.assert_not_called()

    def test_all_tools_are_registered_and_parallel_calls_are_disabled(self):
        client = make_client(response_with_content("普通回答"))
        with patch.object(service, "_get_chat_client", return_value=client):
            service.run_agent_turn(
                TEST_VISITOR_ID,
                "红烧肉怎么做？",
                self.messages,
            )

        call = client.chat.completions.create.call_args
        names = [
            tool["function"]["name"]
            for tool in call.kwargs["tools"]
        ]
        self.assertEqual(
            names,
            [
                service.KNOWLEDGE_BASE_TOOL_NAME,
                service.WEATHER_TOOL_NAME,
                service.POI_TOOL_NAME,
            ],
        )
        self.assertEqual(call.kwargs["tool_choice"], "auto")
        self.assertFalse(call.kwargs["parallel_tool_calls"])

    @patch.object(service, "get_weather", return_value=weather_status("success"))
    @patch.object(service, "knowledge_base_search")
    def test_weather_tool_runs_once_then_second_llm_with_no_sources(
        self,
        search_tool,
        weather_tool,
    ):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"郑州"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            ),
            response_with_content("郑州当前晴，气温 28℃。"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州今天天气怎么样？",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "tool")
        self.assertEqual(result["tool_name"], service.WEATHER_TOOL_NAME)
        self.assertEqual(result["sources"], [])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        weather_tool.assert_called_once_with(location="郑州")
        search_tool.assert_not_called()

        second_messages = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        tool_message = next(
            message for message in second_messages if message.get("role") == "tool"
        )
        tool_payload = json.loads(tool_message["content"])
        self.assertEqual(tool_payload, weather_status("success"))

    @patch.object(service, "get_weather")
    def test_weather_dispatcher_accepts_only_location(self, weather_tool):
        weather_tool.return_value = weather_status("success")

        result = service.execute_tool_call(
            service.WEATHER_TOOL_NAME,
            {"location": "  郑州  "},
            TEST_VISITOR_ID,
        )

        self.assertEqual(result["status"], "success")
        weather_tool.assert_called_once_with(location="郑州")

    @patch.object(service, "get_weather")
    def test_weather_extra_argument_is_rejected(self, weather_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"郑州","units":"metric"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            )
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州天气怎么样？",
                self.messages,
            )

        self.assertEqual(result["status"], "invalid_tool_arguments")
        weather_tool.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 1)

    @patch.object(
        service,
        "get_weather",
        return_value=weather_status("temporarily_unavailable"),
    )
    def test_weather_unavailable_cannot_be_replaced_with_fake_weather(
        self,
        weather_tool,
    ):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"郑州"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            ),
            response_with_content("郑州现在晴，温度 30℃。"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州天气怎么样？",
                self.messages,
            )

        self.assertEqual(result["answer"], "天气服务暂时不可用，请稍后重试。")
        self.assertNotIn("30", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        weather_tool.assert_called_once()

    @patch.object(
        service,
        "get_weather",
        return_value=weather_status("invalid_request"),
    )
    def test_invalid_weather_location_has_controlled_answer(self, weather_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"不存在的地点"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            ),
            response_with_content("这个地点当前气温 20℃。"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "不存在的地点天气怎么样？",
                self.messages,
            )

        self.assertEqual(
            result["answer"],
            "无法识别该地点，请提供更明确的城市或地区名称。",
        )
        self.assertNotIn("20", result["answer"])
        self.assertEqual(result["tool_status"], "invalid_request")
        weather_tool.assert_called_once()

    @patch.object(service, "knowledge_base_search")
    def test_general_rag_question_still_uses_auto(self, search_tool):
        client = make_client(response_with_content("RAG 是检索增强生成。"))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "什么是RAG？",
                self.messages,
            )

        self.assertEqual(result["mode"], "direct")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["tool_choice"],
            "auto",
        )
        search_tool.assert_not_called()

    def test_explicit_knowledge_base_phrases_are_narrowly_detected(self):
        self.assertTrue(service.should_force_knowledge_base("根据我的知识库，什么是RAG？"))
        self.assertTrue(service.should_force_knowledge_base("根据我上传的PDF解释RAG"))
        self.assertTrue(service.should_force_knowledge_base("我的PDF里有没有部署说明？"))
        self.assertFalse(service.should_force_knowledge_base("什么是RAG？"))
        self.assertFalse(service.should_force_knowledge_base("介绍一下AI"))

    def test_explicit_current_weather_is_narrowly_detected(self):
        self.assertTrue(service.should_force_weather("郑州今天天气怎么样？"))
        self.assertTrue(service.should_force_weather("上海现在多少度？"))
        self.assertTrue(service.should_force_weather("杭州今天冷不冷？"))
        self.assertFalse(service.should_force_weather("什么是天气？"))
        self.assertFalse(service.should_force_weather("为什么夏天很热？"))
        self.assertFalse(service.should_force_weather("RAG和天气有什么关系？"))
        self.assertFalse(service.should_force_weather("今天天气怎么样？"))
        self.assertFalse(service.should_force_weather("郑州明天天气怎么样？"))

    def test_future_forecast_is_declined_without_current_weather_tool(self):
        with (
            patch.object(service, "_get_chat_client") as chat_client,
            patch.object(service, "get_weather") as weather_tool,
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州明天天气怎么样？",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "direct")
        self.assertIn("暂不支持未来天气预报", result["answer"])
        chat_client.assert_not_called()
        weather_tool.assert_not_called()

    def test_current_weather_without_location_requests_clarification(self):
        with (
            patch.object(service, "_get_chat_client") as chat_client,
            patch.object(service, "get_weather") as weather_tool,
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "今天天气怎么样？",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "请提供要查询的城市或地区名称。")
        chat_client.assert_not_called()
        weather_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_explicit_knowledge_base_request_forces_named_tool(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"什么是 RAG？"}')),
            response_with_content("RAG 是检索增强生成。[来源1]"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库，什么是RAG？",
                self.messages,
            )

        first_call = client.chat.completions.create.call_args_list[0]
        self.assertEqual(
            first_call.kwargs["tool_choice"],
            service.FORCED_KNOWLEDGE_BASE_TOOL_CHOICE,
        )
        self.assertEqual(result["mode"], "tool")
        search_tool.assert_called_once()

    @patch.object(service, "get_weather")
    @patch.object(service, "knowledge_base_search")
    def test_forced_knowledge_base_request_rejects_weather_tool(
        self,
        search_tool,
        weather_tool,
    ):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"郑州"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            )
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库解释RAG",
                self.messages,
            )

        self.assertEqual(result["status"], "forced_tool_mismatch")
        search_tool.assert_not_called()
        weather_tool.assert_not_called()

    def test_knowledge_base_force_has_priority_over_weather_force(self):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"郑州今天天气"}')),
            response_with_content("知识库中的天气资料。[来源1]"),
        )
        with (
            patch.object(service, "_get_chat_client", return_value=client),
            patch.object(
                service,
                "knowledge_base_search",
                return_value=result_status("results"),
            ) as search_tool,
            patch.object(service, "get_weather") as weather_tool,
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库，郑州今天天气怎么样？",
                self.messages,
            )

        first_call = client.chat.completions.create.call_args_list[0]
        self.assertEqual(
            first_call.kwargs["tool_choice"],
            service.FORCED_KNOWLEDGE_BASE_TOOL_CHOICE,
        )
        self.assertEqual(result["tool_name"], service.KNOWLEDGE_BASE_TOOL_NAME)
        search_tool.assert_called_once()
        weather_tool.assert_not_called()

    def test_forced_weather_direct_response_is_rejected(self):
        client = make_client(response_with_content("我无法获取实时天气。"))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州今天天气怎么样？",
                self.messages,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "forced_tool_not_called")
        self.assertNotIn("无法获取实时天气", result["answer"])
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["tool_choice"],
            service.FORCED_WEATHER_TOOL_CHOICE,
        )

    @patch.object(service, "get_weather")
    @patch.object(service, "knowledge_base_search")
    def test_forced_weather_rejects_knowledge_base_tool(
        self,
        search_tool,
        weather_tool,
    ):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"郑州天气"}'))
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州今天天气怎么样？",
                self.messages,
            )

        self.assertEqual(result["status"], "forced_tool_mismatch")
        search_tool.assert_not_called()
        weather_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_uploaded_pdf_request_forces_named_tool(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"解释 RAG"}')),
            response_with_content("解释结果。[来源1]"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我上传的PDF解释RAG",
                self.messages,
            )

        self.assertEqual(
            client.chat.completions.create.call_args_list[0].kwargs["tool_choice"],
            service.FORCED_KNOWLEDGE_BASE_TOOL_CHOICE,
        )
        search_tool.assert_called_once()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_tool_answer_calls_llm_twice_tool_once_and_returns_sources(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"什么是 RAG？"}')),
            response_with_content("RAG 是检索增强生成。[来源1]"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "什么是 RAG？",
                self.messages,
            )

        self.assertEqual(result["mode"], "tool")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        search_tool.assert_called_once_with(
            visitor_id=TEST_VISITOR_ID,
            query="什么是 RAG？",
        )
        self.assertEqual(
            result["sources"],
            [{"source_file": "知识库.pdf", "page_number": 2}],
        )

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_model_source_list_is_removed_but_inline_citation_is_kept(self, _search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"什么是 RAG？"}')),
            response_with_content(
                "RAG 是检索增强生成。[来源1]\n\n"
                "来源：\n"
                "知识库.pdf · 第2页[来源1]\n"
                "知识库.pdf · 第2页[来源2]"
            ),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "什么是 RAG？",
                self.messages,
            )

        self.assertEqual(result["answer"], "RAG 是检索增强生成。[来源1]")
        self.assertNotIn("\n来源：", result["answer"])
        self.assertEqual(
            result["sources"],
            [{"source_file": "知识库.pdf", "page_number": 2}],
        )

    def test_no_knowledge_base_still_gets_a_safe_final_answer(self):
        self._assert_non_result_tool_status("no_knowledge_base")

    def test_no_relevant_results_still_gets_a_safe_final_answer(self):
        self._assert_non_result_tool_status("no_relevant_results")

    def test_temporarily_unavailable_still_gets_a_safe_final_answer(self):
        self._assert_non_result_tool_status("temporarily_unavailable")

    def test_forced_no_relevant_result_never_returns_general_tutorial(self):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"维修汽车发动机"}')),
            response_with_content("先拆卸发动机，然后检查火花塞。"),
        )
        with (
            patch.object(service, "_get_chat_client", return_value=client),
            patch.object(
                service,
                "knowledge_base_search",
                return_value=result_status("no_relevant_results"),
            ),
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库告诉我怎么维修汽车发动机",
                self.messages,
            )

        self.assertIn("当前知识库中没有找到", result["answer"])
        self.assertNotIn("拆卸发动机", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_forced_request_without_knowledge_base_has_fixed_semantics(self):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"查询资料"}')),
            response_with_content("我可以根据通用知识回答。"),
        )
        with (
            patch.object(service, "_get_chat_client", return_value=client),
            patch.object(
                service,
                "knowledge_base_search",
                return_value=result_status("no_knowledge_base"),
            ),
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "查一下我的知识库，这个问题怎么处理？",
                self.messages,
            )

        self.assertEqual(
            result["answer"],
            "你目前还没有建立知识库，无法根据知识库回答。",
        )
        self.assertNotIn("通用知识", result["answer"])
        self.assertEqual(result["sources"], [])

    def _assert_non_result_tool_status(self, status: str):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"测试问题"}')),
            response_with_content("根据当前情况给出普通回答。"),
        )
        with (
            patch.object(service, "_get_chat_client", return_value=client),
            patch.object(
                service,
                "knowledge_base_search",
                return_value=result_status(status),
            ) as search_tool,
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "tool")
        self.assertEqual(result["tool_status"], status)
        self.assertEqual(result["sources"], [])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        search_tool.assert_called_once()

    @patch.object(service, "knowledge_base_search")
    def test_invalid_json_arguments_are_rejected(self, search_tool):
        client = make_client(response_with_tool_calls(make_tool_call("{broken")))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "invalid_tool_arguments")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        search_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search")
    def test_unknown_tool_is_rejected(self, search_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call('{"query":"测试"}', name="unknown_tool")
            )
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertEqual(result["status"], "unknown_tool")
        search_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_second_model_tool_call_is_not_executed(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"第一次"}')),
            response_with_tool_calls(
                make_tool_call('{"query":"第二次"}', call_id="call-2")
            ),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertEqual(result["status"], "repeated_tool_call")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        search_tool.assert_called_once()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_visitor_id_is_injected_by_server_not_model(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"查询资料"}')),
            response_with_content("回答"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            service.run_agent_turn(
                OTHER_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        search_tool.assert_called_once_with(
            visitor_id=OTHER_VISITOR_ID,
            query="查询资料",
        )
        schema_text = json.dumps(
            service.KNOWLEDGE_BASE_SEARCH_TOOL,
            ensure_ascii=False,
        )
        self.assertNotIn("visitor_id", schema_text)

    @patch.object(service, "knowledge_base_search")
    def test_model_supplied_visitor_id_is_rejected(self, search_tool):
        arguments = json.dumps(
            {"query": "查询资料", "visitor_id": OTHER_VISITOR_ID}
        )
        client = make_client(response_with_tool_calls(make_tool_call(arguments)))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertEqual(result["status"], "invalid_tool_arguments")
        search_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_original_messages_unchanged_and_tool_messages_are_temporary(self, _search_tool):
        original = copy.deepcopy(self.messages)
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"什么是 RAG？"}')),
            response_with_content("最终回答"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            service.run_agent_turn(
                TEST_VISITOR_ID,
                "什么是 RAG？",
                self.messages,
            )

        self.assertEqual(self.messages, original)
        second_messages = client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        self.assertTrue(any(message.get("tool_calls") for message in second_messages))
        self.assertTrue(any(message.get("role") == "tool" for message in second_messages))
        self.assertFalse(any(message.get("role") == "tool" for message in self.messages))

    @patch.object(service, "search_poi", return_value=poi_status())
    def test_poi_city_search_arguments_are_dispatched(self, poi_tool):
        result = service.execute_tool_call(
            service.POI_TOOL_NAME,
            '{"query":"  咖啡店  ","city":"  郑州  "}',
            TEST_VISITOR_ID,
        )

        self.assertEqual(result["status"], "results")
        poi_tool.assert_called_once_with(
            query="咖啡店",
            city="郑州",
            anchor=None,
        )

    @patch.object(service, "search_poi", return_value=poi_status())
    def test_poi_nearby_arguments_include_optional_anchor(self, poi_tool):
        service.execute_tool_call(
            service.POI_TOOL_NAME,
            {"query": "餐厅", "city": "北京", "anchor": "故宫"},
            TEST_VISITOR_ID,
        )

        poi_tool.assert_called_once_with(
            query="餐厅",
            city="北京",
            anchor="故宫",
        )

    def test_poi_missing_query_is_rejected(self):
        with self.assertRaises(service.ToolDispatchError) as caught:
            service.execute_tool_call(
                service.POI_TOOL_NAME,
                {"city": "郑州"},
                TEST_VISITOR_ID,
            )
        self.assertEqual(caught.exception.status, "invalid_tool_arguments")

    def test_poi_missing_city_is_rejected(self):
        with self.assertRaises(service.ToolDispatchError) as caught:
            service.execute_tool_call(
                service.POI_TOOL_NAME,
                {"query": "咖啡店"},
                TEST_VISITOR_ID,
            )
        self.assertEqual(caught.exception.status, "invalid_tool_arguments")

    @patch.object(service, "search_poi")
    def test_poi_unknown_or_server_fields_are_rejected(self, poi_tool):
        forbidden_arguments = (
            {"query": "餐厅", "city": "北京", "radius": 3000},
            {"query": "餐厅", "city": "北京", "visitor_id": OTHER_VISITOR_ID},
            {"query": "餐厅", "city": "北京", "provider": "amap"},
        )
        for arguments in forbidden_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(service.ToolDispatchError) as caught:
                    service.execute_tool_call(
                        service.POI_TOOL_NAME,
                        arguments,
                        TEST_VISITOR_ID,
                    )
                self.assertEqual(caught.exception.status, "invalid_tool_arguments")
        poi_tool.assert_not_called()

    @patch.object(service, "search_poi", return_value=poi_status())
    def test_poi_agent_turn_uses_temporary_tool_result_and_no_sources(self, poi_tool):
        original_messages = copy.deepcopy(self.messages)
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"query":"餐厅","city":"北京","anchor":"故宫"}',
                    name=service.POI_TOOL_NAME,
                )
            ),
            response_with_content("故宫附近可以考虑景运门故宫餐厅。"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "北京故宫附近有什么餐厅？",
                self.messages,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "tool")
        self.assertEqual(result["tool_name"], service.POI_TOOL_NAME)
        self.assertEqual(result["sources"], [])
        poi_tool.assert_called_once_with(query="餐厅", city="北京", anchor="故宫")
        self.assertEqual(self.messages, original_messages)

        first_call, second_call = client.chat.completions.create.call_args_list
        self.assertEqual(first_call.kwargs["tool_choice"], "auto")
        self.assertFalse(first_call.kwargs["parallel_tool_calls"])
        self.assertEqual(second_call.kwargs["tool_choice"], "none")
        tool_message = next(
            message
            for message in second_call.kwargs["messages"]
            if message.get("role") == "tool"
        )
        self.assertEqual(json.loads(tool_message["content"]), poi_status())
        self.assertFalse(any(message.get("role") == "tool" for message in self.messages))

    @patch.object(service, "search_poi", return_value=poi_status())
    def test_poi_results_create_display_data(self, _poi_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"query":"餐厅","city":"北京","anchor":"故宫"}',
                    name=service.POI_TOOL_NAME,
                )
            ),
            response_with_content("推荐回答"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "北京故宫附近有什么餐厅？",
                self.messages,
            )

        display_data = result["display_data"]
        self.assertEqual(display_data["type"], "poi_results")
        self.assertEqual(display_data["query"], "餐厅")
        self.assertEqual(display_data["city"], "北京")
        self.assertEqual(display_data["search_mode"], "nearby")
        self.assertEqual(display_data["anchor"], "故宫")
        self.assertEqual(len(display_data["items"]), 1)

    def test_poi_display_data_uses_strict_item_allowlist_and_limit(self):
        payload = poi_status()
        base_item = payload["results"][0]
        payload["results"] = []
        for index in range(7):
            item = copy.deepcopy(base_item)
            item["rank"] = index + 1
            item["name"] = f"地点{index + 1}"
            item["api_key"] = "fake-api-key-that-must-not-leak"
            item["provider_raw"] = {"infocode": "10000"}
            item["request_url"] = "https://provider.invalid/sensitive"
            payload["results"].append(item)

        display_data = service._build_poi_display_data(payload)

        self.assertEqual(len(display_data["items"]), 5)
        allowed = service.POI_DISPLAY_ITEM_FIELDS
        for item in display_data["items"]:
            self.assertTrue(set(item).issubset(allowed))
        serialized = json.dumps(display_data, ensure_ascii=False)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("provider_raw", serialized)
        self.assertNotIn("request_url", serialized)
        self.assertNotIn("infocode", serialized)
        self.assertNotIn("fake-api-key", serialized)

    def test_poi_display_location_has_only_coordinates(self):
        payload = poi_status()
        payload["results"][0]["location"].update(
            {
                "formatted_address": "不得返回",
                "provider_id": "secret",
            }
        )

        display_data = service._build_poi_display_data(payload)
        location = display_data["items"][0]["location"]

        self.assertEqual(set(location), {"longitude", "latitude"})
        self.assertEqual(
            set(location),
            service.POI_DISPLAY_LOCATION_FIELDS,
        )

    def test_poi_display_ignores_invalid_field_types(self):
        payload = poi_status()
        item = payload["results"][0]
        item.update(
            {
                "distance_m": "142",
                "rating": float("nan"),
                "cost_per_person": -1,
                "tags": ["中餐", 42, "  "],
                "location": {
                    "longitude": 999,
                    "latitude": "39.9",
                },
            }
        )

        display_item = service._build_poi_display_data(payload)["items"][0]

        self.assertNotIn("distance_m", display_item)
        self.assertNotIn("rating", display_item)
        self.assertNotIn("cost_per_person", display_item)
        self.assertEqual(display_item["tags"], ["中餐"])
        self.assertNotIn("location", display_item)

    def test_poi_non_result_statuses_have_no_display_data(self):
        for status in (
            "no_results",
            "location_required",
            "ambiguous_location",
            "temporarily_unavailable",
        ):
            with self.subTest(status=status):
                self.assertIsNone(
                    service._build_poi_display_data(poi_status(status))
                )

    def test_poi_malformed_results_do_not_break_display_builder(self):
        malformed_values = (
            None,
            [],
            {"status": "results", "results": None},
            {"status": "results", "results": [None, "bad"]},
            {"status": "results", "results": [{"name": ""}]},
        )
        for value in malformed_values:
            with self.subTest(value=value):
                self.assertIsNone(service._build_poi_display_data(value))

    @patch.object(service, "get_weather", return_value=weather_status("success"))
    def test_weather_result_has_no_display_data(self, _weather_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"location":"郑州"}',
                    name=service.WEATHER_TOOL_NAME,
                )
            ),
            response_with_content("郑州今天晴。"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州今天天气怎么样？",
                self.messages,
            )
        self.assertIsNone(result["display_data"])

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_knowledge_base_result_has_no_display_data(self, _search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"RAG"}')),
            response_with_content("RAG 回答"),
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库，什么是 RAG？",
                self.messages,
            )
        self.assertIsNone(result["display_data"])

    def test_direct_result_has_no_display_data(self):
        client = make_client(response_with_content("普通回答"))
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "红烧肉怎么做？",
                self.messages,
            )
        self.assertIsNone(result["display_data"])

    def test_document_sources_deduplicate_by_real_locator(self):
        tool_result = {
            "status": "results",
            "results": [
                {
                    "source_file": "旅行计划.md",
                    "source_type": "md",
                    "page_number": None,
                    "locator_type": "section",
                    "locator_value": "交通安排",
                },
                {
                    "source_file": "旅行计划.md",
                    "source_type": "md",
                    "page_number": None,
                    "locator_type": "section",
                    "locator_value": "交通安排",
                },
                {
                    "source_file": "旅行计划.md",
                    "source_type": "md",
                    "page_number": None,
                    "locator_type": "section",
                    "locator_value": "餐饮建议",
                },
            ],
        }

        sources = service._deduplicated_sources(tool_result)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source_type"], "md")
        self.assertIsNone(sources[0]["page_number"])
        self.assertEqual(sources[0]["locator_value"], "交通安排")

    def test_low_score_pdf_candidates_never_reach_agent_sources(self):
        raw_results = [
            {
                "score": 0.62,
                "chunk_text": "城市生活助手测试口令是蓝鲸7392。",
                "source_file": "测试.docx",
                "source_type": "docx",
                "page_number": None,
                "locator_type": "paragraph",
                "locator_value": "1",
                "document_id": "docx-document",
                "chunk_id": "docx-chunk",
            },
            *[
                {
                    "score": score,
                    "chunk_text": f"无关旧 PDF 内容 {index}",
                    "source_file": f"旧资料{index}.pdf",
                    "source_type": "pdf",
                    "page_number": index,
                    "locator_type": "page",
                    "locator_value": str(index),
                    "document_id": f"pdf-document-{index}",
                    "chunk_id": f"pdf-chunk-{index}",
                }
                for index, score in enumerate((0.31, 0.27, 0.19), start=1)
            ],
        ]
        with (
            patch.object(kb_tool, "has_knowledge_base", return_value=True),
            patch.object(kb_tool, "embed_text", return_value=[1.0]),
            patch.object(kb_tool, "search", return_value=raw_results),
        ):
            filtered_tool_result = kb_tool.knowledge_base_search(
                TEST_VISITOR_ID,
                "城市生活助手测试口令是什么？",
            )

        self.assertEqual(filtered_tool_result["result_count"], 1)
        self.assertEqual(
            [item["source_file"] for item in filtered_tool_result["results"]],
            ["测试.docx"],
        )
        self.assertNotIn("旧资料", json.dumps(filtered_tool_result, ensure_ascii=False))

        client = make_client(
            response_with_tool_calls(
                make_tool_call('{"query":"城市生活助手测试口令"}')
            ),
            response_with_content("测试口令是蓝鲸7392。"),
        )
        with (
            patch.object(
                service,
                "knowledge_base_search",
                return_value=filtered_tool_result,
            ),
            patch.object(service, "_get_chat_client", return_value=client),
        ):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "根据我的知识库，城市生活助手测试口令是什么？",
                self.messages,
            )

        self.assertEqual(result["mode"], "tool")
        self.assertEqual(
            result["sources"],
            [
                {
                    "source_file": "测试.docx",
                    "page_number": None,
                    "source_type": "docx",
                    "locator_type": "paragraph",
                    "locator_value": "1",
                }
            ],
        )
        self.assertNotIn("旧资料", json.dumps(result, ensure_ascii=False))

    @patch.object(service, "search_poi", side_effect=RuntimeError("secret key and path"))
    def test_poi_tool_exception_is_safely_returned(self, poi_tool):
        client = make_client(
            response_with_tool_calls(
                make_tool_call(
                    '{"query":"咖啡店","city":"郑州"}',
                    name=service.POI_TOOL_NAME,
                )
            )
        )
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "郑州有什么咖啡店？",
                self.messages,
            )

        self.assertEqual(result["status"], "tool_execution_failed")
        self.assertNotIn("secret", str(result))
        poi_tool.assert_called_once()

    def test_poi_argument_schema_declares_required_and_optional_fields(self):
        schema = service.TOOL_ARGUMENT_SCHEMAS[service.POI_TOOL_NAME]
        self.assertEqual(schema["allowed_fields"], {"query", "city", "anchor"})
        self.assertEqual(schema["required_fields"], {"query", "city"})
        self.assertEqual(schema["optional_fields"], {"anchor"})

    @patch.object(service, "knowledge_base_search")
    def test_first_model_failure_is_sanitized(self, search_tool):
        client = make_client()
        client.chat.completions.create.side_effect = RuntimeError("secret path/key")
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertEqual(result["status"], "first_model_call_failed")
        self.assertNotIn("secret", str(result))
        search_tool.assert_not_called()

    @patch.object(service, "knowledge_base_search", return_value=result_status("results"))
    def test_second_model_failure_is_sanitized(self, search_tool):
        client = make_client(
            response_with_tool_calls(make_tool_call('{"query":"测试"}')),
        )
        client.chat.completions.create.side_effect = [
            response_with_tool_calls(make_tool_call('{"query":"测试"}')),
            RuntimeError("secret path/key"),
        ]
        with patch.object(service, "_get_chat_client", return_value=client):
            result = service.run_agent_turn(
                TEST_VISITOR_ID,
                "测试问题",
                self.messages,
            )

        self.assertEqual(result["status"], "second_model_call_failed")
        self.assertNotIn("secret", str(result))
        search_tool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
