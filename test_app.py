"""
test_app.py — Full pytest suite for the Groq-powered insurance RAG app.

Run with:
    pytest test_app.py -v
"""

import sys
import json
import os
from unittest.mock import MagicMock

# ── MUST happen BEFORE importing app ─────────────────────────────────────────
# Mock streamlit so module-level st.set_page_config / st.markdown don't crash.
_st_mock = MagicMock()
sys.modules["streamlit"] = _st_mock

# Mock spacy so QueryParser doesn't crash if en_core_web_sm is absent.
_spacy_mock = MagicMock()
sys.modules["spacy"] = _spacy_mock

import pytest

# Now import app — all st.* calls silently hit the mock.
import app

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model name constant
# ─────────────────────────────────────────────────────────────────────────────
class TestModelNameConstant:
    def test_model_is_groq_llama_3_3_70b(self):
        assert app.MODEL_NAME == "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# 2. clean_json_response
# ─────────────────────────────────────────────────────────────────────────────
class TestCleanJsonResponse:
    def test_strips_line_comments(self):
        raw = '{"key": "value"} // this is a comment\n'
        cleaned = app.clean_json_response(raw)
        assert "//" not in cleaned
        assert '"key": "value"' in cleaned

    def test_strips_block_comments(self):
        raw = '{"key": /* a block comment */ "value"}'
        cleaned = app.clean_json_response(raw)
        assert "/*" not in cleaned
        assert "*/" not in cleaned

    def test_trailing_comma_before_brace_is_valid_json_after_clean(self):
        raw = '{"key": "value",}'
        cleaned = app.clean_json_response(raw)
        parsed = json.loads(cleaned)
        assert parsed["key"] == "value"

    def test_trailing_comma_before_bracket_is_valid_json_after_clean(self):
        raw = '{"items": [1, 2, 3,]}'
        cleaned = app.clean_json_response(raw)
        parsed = json.loads(cleaned)
        assert parsed["items"] == [1, 2, 3]

    def test_noop_on_already_clean_json(self):
        raw = '{"decision": "APPROVED", "amount": 50000}'
        cleaned = app.clean_json_response(raw)
        assert json.loads(cleaned)["decision"] == "APPROVED"


# ─────────────────────────────────────────────────────────────────────────────
# 3. QueryParser
# ─────────────────────────────────────────────────────────────────────────────
class TestQueryParser:
    def setup_method(self):
        self.qp = app.QueryParser()

    def test_age_extraction(self):
        result = self.qp.parse_query("35-year-old male with knee surgery")
        assert result["age"] == 35

    def test_gender_extraction_male(self):
        result = self.qp.parse_query("35-year-old male with knee surgery")
        assert result["gender"] == "male"

    def test_gender_extraction_female(self):
        result = self.qp.parse_query("28-year-old female, maternity claim")
        assert result["gender"] == "female"

    def test_condition_extraction(self):
        result = self.qp.parse_query("knee surgery for 40-year-old male")
        assert result["condition"] is not None
        cond = result["condition"]
        assert "knee" in cond or "surgery" in cond

    def test_amount_extraction_rupee_symbol(self):
        result = self.qp.parse_query("claim for ₹50,000 treatment")
        assert result["amount_mentioned"] == 50000

    def test_policy_duration_not_confused_with_age(self):
        result = self.qp.parse_query("3-year-old policy, 45-year-old male, cardiac surgery")
        assert result["age"] == 45
        assert result["policy_duration"] == 3

    def test_missing_fields_all_empty(self):
        result = self.qp.parse_query("claim request")
        missing = self.qp.get_missing_fields(result)
        assert "age" in missing
        assert "gender" in missing
        assert "medical condition/treatment" in missing

    def test_missing_fields_partial_when_age_and_gender_set(self):
        result = self.qp.parse_query("45-year-old male")
        missing = self.qp.get_missing_fields(result)
        assert "age" not in missing
        assert "gender" not in missing


# ─────────────────────────────────────────────────────────────────────────────
# 4. InsuranceRuleEngine
# ─────────────────────────────────────────────────────────────────────────────
class TestInsuranceRuleEngine:
    def setup_method(self):
        self.engine = app.InsuranceRuleEngine()
        self.engine.extracted_rules = {}  # always start from defaults

    def test_extract_pre_existing_waiting_period_months(self):
        rules = self.engine.extract_rules_from_policy(
            "Pre-existing conditions have a waiting period of 24 months."
        )
        assert "waiting_periods" in rules
        assert rules["waiting_periods"].get("pre_existing") == 24

    def test_extract_maternity_waiting_period(self):
        rules = self.engine.extract_rules_from_policy(
            "Maternity benefits require a 9-month waiting period."
        )
        assert rules.get("waiting_periods", {}).get("maternity") == 9

    def test_extract_age_limits_range(self):
        rules = self.engine.extract_rules_from_policy(
            "Entry age is 18 to 65 years."
        )
        assert "age_limits" in rules
        assert rules["age_limits"]["entry_age"]["min"] == 18
        assert rules["age_limits"]["entry_age"]["max"] == 65

    def test_validate_age_above_max_fails(self):
        # Default max entry age = 65
        parsed = {
            "age": 70, "gender": "male", "condition": "surgery",
            "policy_duration": 24, "policy_duration_unit": "month",
            "completeness_score": 0.5
        }
        result = self.engine.validate_claim(parsed)
        assert result["passed"] is False
        assert any("70" in v for v in result["violations"])

    def test_validate_pre_existing_waiting_period_violated(self):
        # Default pre_existing waiting = 24 months; policy only 6 months old
        parsed = {
            "age": 40, "gender": "male", "condition": "diabetes",
            "policy_duration": 6, "policy_duration_unit": "month",
            "completeness_score": 0.6
        }
        result = self.engine.validate_claim(parsed)
        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_validate_pre_existing_waiting_period_satisfied(self):
        # Policy 30 months old > 24-month default waiting period
        parsed = {
            "age": 40, "gender": "male", "condition": "diabetes",
            "policy_duration": 30, "policy_duration_unit": "month",
            "completeness_score": 0.7
        }
        result = self.engine.validate_claim(parsed)
        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_validate_no_condition_passes(self):
        parsed = {
            "age": 30, "gender": "female", "condition": None,
            "policy_duration": None, "policy_duration_unit": None,
            "completeness_score": 0.2
        }
        result = self.engine.validate_claim(parsed)
        assert result["passed"] is True

    def test_validate_maternity_waiting_violated(self):
        # Default maternity waiting = 9 months; policy 3 months old
        parsed = {
            "age": 28, "gender": "female", "condition": "maternity",
            "policy_duration": 3, "policy_duration_unit": "month",
            "completeness_score": 0.6
        }
        result = self.engine.validate_claim(parsed)
        assert result["passed"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. ConfidenceCalculator
# ─────────────────────────────────────────────────────────────────────────────
class TestConfidenceCalculator:
    def setup_method(self):
        self.calc = app.ConfidenceCalculator()

    @staticmethod
    def _make_doc(text: str):
        doc = MagicMock()
        doc.page_content = text
        return doc

    def test_all_zeros_gives_minimum_confidence(self):
        # When all meaningful inputs are zero the only contributor is
        # response_consistency (non-zero even for an empty justification).
        # The resulting floor is ~0.05; assert it is very low (<= 0.1).
        parsed = {"completeness_score": 0.0}
        rule_val = {"confidence_impact": 0.0, "violations": []}
        llm_resp = {"confidence": 0.0, "decision": "APPROVED", "amount": 0, "justification": ""}
        result = self.calc.calculate_comprehensive_confidence(
            "query", parsed, [], rule_val, llm_resp
        )
        assert result["final_confidence"] <= 0.1, (
            f"Expected near-zero confidence, got {result['final_confidence']}"
        )

    def test_high_inputs_give_high_confidence(self):
        parsed = {"completeness_score": 1.0}
        rule_val = {"confidence_impact": 0.3, "violations": []}
        llm_resp = {
            "confidence": 1.0,
            "decision": "APPROVED",
            "amount": 50000,
            "justification": "Covered under section 3.1 on page 5 of the policy document",
        }
        docs = [self._make_doc("knee surgery treatment approved covered policy")]
        result = self.calc.calculate_comprehensive_confidence(
            "knee surgery claim", parsed, docs, rule_val, llm_resp
        )
        assert result["final_confidence"] > 0.7

    def test_violations_reduce_confidence_vs_clean(self):
        parsed = {"completeness_score": 0.8}
        llm_resp = {
            "confidence": 0.8,
            "decision": "APPROVED",
            "amount": 50000,
            "justification": "Covered under policy section 2 on page 3",
        }
        docs = [self._make_doc("general insurance claim")]

        clean = self.calc.calculate_comprehensive_confidence(
            "test", parsed, docs, {"confidence_impact": 0.0, "violations": []}, llm_resp
        )
        violated = self.calc.calculate_comprehensive_confidence(
            "test", parsed, docs, {"confidence_impact": -0.5, "violations": ["Age exceeded"]}, llm_resp
        )
        assert violated["final_confidence"] < clean["final_confidence"]

    def test_response_consistency_rejected_with_nonzero_amount_is_penalised(self):
        score = self.calc.calculate_response_consistency(
            {"decision": "REJECTED", "amount": 50000, "justification": "some justification here"}
        )
        assert score < 1.0

    def test_response_consistency_approved_with_zero_amount_is_penalised(self):
        score = self.calc.calculate_response_consistency(
            {"decision": "APPROVED", "amount": 0, "justification": "Covered under section 3.1"}
        )
        assert score < 1.0

    def test_result_contains_required_keys(self):
        parsed = {"completeness_score": 0.5}
        result = self.calc.calculate_comprehensive_confidence(
            "query", parsed, [],
            {"confidence_impact": 0.0, "violations": []},
            {"confidence": 0.5, "decision": "APPROVED", "amount": 10000,
             "justification": "see page 1 for details"}
        )
        for key in ("final_confidence", "factor_breakdown", "weights_used", "explanation"):
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. process_enhanced_response (parsing paths)
# ─────────────────────────────────────────────────────────────────────────────
class TestResponseParsing:
    _VALID = json.dumps({
        "decision": "APPROVED",
        "amount": 50000,
        "justification": "Covered under section 2.1 on page 3",
        "confidence": 0.9,
        "reasoning_steps": ["Step 1: Check coverage", "Step 2: Confirm amount"],
        "source_pages": [3, 5],
        "rule_violations": [],
    })

    @staticmethod
    def _no_violations():
        return {"violations": [], "confidence_impact": 0.0}

    @staticmethod
    def _failing_parser():
        p = MagicMock()
        p.parse.side_effect = Exception("forced parse failure")
        return p

    @staticmethod
    def _succeeding_parser(return_value):
        p = MagicMock()
        p.parse.return_value = return_value
        return p

    def test_structured_path(self):
        expected = json.loads(self._VALID)
        parser = self._succeeding_parser(expected)
        result, method = app.process_enhanced_response(self._VALID, parser, self._no_violations())
        assert method == "structured"
        assert result["decision"] == "APPROVED"
        assert result["amount"] == 50000

    def test_direct_json_fallback(self):
        result, method = app.process_enhanced_response(
            self._VALID, self._failing_parser(), self._no_violations()
        )
        assert method == "direct_json"
        assert result["decision"] == "APPROVED"

    def test_json_with_inline_comments(self):
        raw = (
            '{"decision": "APPROVED", // inline comment\n'
            '"amount": 30000, "justification": "see page 2", '
            '"confidence": 0.8, "reasoning_steps": [], "source_pages": [2], "rule_violations": []}'
        )
        result, method = app.process_enhanced_response(
            raw, self._failing_parser(), self._no_violations()
        )
        assert result["decision"] == "APPROVED"
        assert result["amount"] == 30000

    def test_json_with_trailing_comma(self):
        raw = (
            '{"decision": "REJECTED", "amount": 0, '
            '"justification": "waiting period page 1", "confidence": 0.85, '
            '"reasoning_steps": [], "source_pages": [1], "rule_violations": [],}'
        )
        result, method = app.process_enhanced_response(
            raw, self._failing_parser(), self._no_violations()
        )
        assert result["decision"] == "REJECTED"

    def test_json_extraction_from_surrounding_text(self):
        raw = (
            'Here is the analysis:\n'
            '{"decision": "REQUIRES_CLARIFICATION", "amount": 0, '
            '"justification": "need more info", "confidence": 0.4, '
            '"reasoning_steps": [], "source_pages": [], "rule_violations": []}\n'
            'End of response.'
        )
        result, method = app.process_enhanced_response(
            raw, self._failing_parser(), self._no_violations()
        )
        assert method == "json_extraction"
        assert result["decision"] == "REQUIRES_CLARIFICATION"

    def test_completely_broken_input_returns_error(self):
        result, method = app.process_enhanced_response(
            "this is totally unparseable text without any json structure",
            self._failing_parser(),
            self._no_violations(),
        )
        assert method == "error"
        assert result["decision"] == "ERROR"

    def test_missing_fields_are_defaulted(self):
        # JSON with only decision; all other fields must be defaulted
        result, _ = app.process_enhanced_response(
            '{"decision": "APPROVED"}',
            self._failing_parser(),
            self._no_violations(),
        )
        for field in ("amount", "confidence", "reasoning_steps", "source_pages", "justification"):
            assert field in result, f"Missing expected field: {field}"

    def test_confidence_normalised_from_string(self):
        raw = json.dumps({
            "decision": "APPROVED", "amount": 10000,
            "justification": "see page 4", "confidence": "0.75",
            "reasoning_steps": [], "source_pages": [4], "rule_violations": [],
        })
        result, _ = app.process_enhanced_response(
            raw, self._failing_parser(), self._no_violations()
        )
        assert isinstance(result["confidence"], float)
        assert result["confidence"] == pytest.approx(0.75)

    def test_string_amount_with_comma_normalised_to_int(self):
        raw = json.dumps({
            "decision": "APPROVED", "amount": "50,000",
            "justification": "see page 1", "confidence": 0.8,
            "reasoning_steps": [], "source_pages": [1], "rule_violations": [],
        })
        result, _ = app.process_enhanced_response(
            raw, self._failing_parser(), self._no_violations()
        )
        assert result["amount"] == 50000


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rule override logic
# ─────────────────────────────────────────────────────────────────────────────
class TestRuleOverride:
    @staticmethod
    def _failing_parser():
        p = MagicMock()
        p.parse.side_effect = Exception("forced fallback")
        return p

    def test_approved_overridden_to_rejected_when_violation_present(self):
        raw = json.dumps({
            "decision": "APPROVED", "amount": 80000,
            "justification": "Surgery covered on page 5",
            "confidence": 0.9,
            "reasoning_steps": ["Step 1", "Step 2"],
            "source_pages": [5], "rule_violations": [],
        })
        rule_validation = {
            "violations": ["Pre-existing condition waiting period not satisfied"],
            "confidence_impact": -0.4,
        }
        result, _ = app.process_enhanced_response(raw, self._failing_parser(), rule_validation)
        assert result["decision"] == "REJECTED"
        assert result["amount"] == 0
        assert len(result["rule_violations"]) > 0

    def test_rejected_llm_amount_forced_to_zero(self):
        # LLM mistakenly returns amount=50000 for REJECTED — must be zeroed.
        raw = json.dumps({
            "decision": "REJECTED", "amount": 50000,
            "justification": "not covered, see page 2",
            "confidence": 0.8,
            "reasoning_steps": [], "source_pages": [2], "rule_violations": [],
        })
        result, _ = app.process_enhanced_response(
            raw, self._failing_parser(), {"violations": [], "confidence_impact": 0.0}
        )
        assert result["amount"] == 0

    def test_violations_appended_to_rule_violations_field(self):
        raw = json.dumps({
            "decision": "APPROVED", "amount": 30000,
            "justification": "page 3",
            "confidence": 0.85,
            "reasoning_steps": [], "source_pages": [3], "rule_violations": [],
        })
        rule_validation = {
            "violations": ["Age 70 exceeds maximum entry age 65"],
            "confidence_impact": -0.3,
        }
        result, _ = app.process_enhanced_response(raw, self._failing_parser(), rule_validation)
        assert "Age 70 exceeds maximum entry age 65" in result["rule_violations"]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Groq connection (integration — requires real API key)
# ─────────────────────────────────────────────────────────────────────────────
class TestGroqConnection:
    def test_groq_api_key_loaded_from_dotenv(self):
        from dotenv import load_dotenv
        load_dotenv()
        assert os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY not found — check your .env file"

    def test_groq_live_call_returns_non_empty_response(self):
        """Real minimal call to Groq — verifies API key, model name, and connectivity."""
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            pytest.skip("GROQ_API_KEY not set")

        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm = ChatGroq(model=app.MODEL_NAME, temperature=0.0, groq_api_key=api_key)
        response = llm.invoke([HumanMessage(content="Reply with exactly one word: PONG")])
        text = response.content if hasattr(response, "content") else str(response)
        assert len(text.strip()) > 0, "Groq returned an empty response"


# ─────────────────────────────────────────────────────────────────────────────
# 9. BatchProcessor (unit — mocked QA chain)
# ─────────────────────────────────────────────────────────────────────────────
class TestBatchProcessor:
    @staticmethod
    def _make_chain(decision="APPROVED", amount=50000):
        raw = json.dumps({
            "decision": decision, "amount": amount,
            "justification": "Covered under section 1.2 on page 4",
            "confidence": 0.88,
            "reasoning_steps": ["verified coverage", "confirmed amount"],
            "source_pages": [4], "rule_violations": [],
        })
        chain = MagicMock()
        chain.invoke.return_value = {"result": raw, "source_documents": []}
        return chain

    @staticmethod
    def _failing_parser():
        p = MagicMock()
        p.parse.side_effect = Exception("forced json fallback")
        return p

    def setup_method(self):
        self.qp = app.QueryParser()
        self.re = app.InsuranceRuleEngine()
        self.cc = app.ConfidenceCalculator()

    def _processor(self, chain):
        return app.BatchProcessor(
            chain, self._failing_parser(), self.qp, self.re, self.cc, {}
        )

    def test_returns_one_result_per_query(self):
        results = self._processor(self._make_chain()).process_batch([
            "35-year-old male, knee surgery, 2-year policy",
            "28-year-old female, maternity, 1-year policy",
        ])
        assert len(results) == 2

    def test_result_contains_required_schema_keys(self):
        results = self._processor(self._make_chain()).process_batch([
            "40-year-old male, cardiac surgery, policy 3 years old"
        ])
        r = results[0]
        for key in ("decision", "amount", "justification", "confidence",
                    "rule_validation", "parsed_query"):
            assert key in r, f"Missing key in result: {key}"

    def test_chain_error_is_captured_gracefully(self):
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("Simulated network error")
        results = self._processor(chain).process_batch(["any query text"])
        assert results[0]["decision"] == "ERROR"
        assert "error" in results[0]

    def test_batch_index_matches_query_position(self):
        results = self._processor(self._make_chain()).process_batch([
            "query one", "query two", "query three"
        ])
        for i, r in enumerate(results):
            assert r["batch_index"] == i

    def test_rejected_decision_has_zero_amount(self):
        results = self._processor(self._make_chain(decision="REJECTED", amount=0)).process_batch([
            "diabetes patient, 3-month policy"
        ])
        assert results[0]["decision"] == "REJECTED"
        assert results[0]["amount"] == 0

    def test_error_entry_has_all_required_fields(self):
        """Batch error entries must include all analytics-required fields."""
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("boom")
        results = self._processor(chain).process_batch(["any query"])
        r = results[0]
        for key in ("decision", "amount", "confidence", "rule_violations",
                     "parsed_query", "timestamp"):
            assert key in r, f"Error entry missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Patch: clean_json_response preserves URLs inside strings
# ─────────────────────────────────────────────────────────────────────────────
class TestCleanJsonPreservesURLs:
    def test_url_inside_string_value_is_preserved(self):
        raw = '{"justification": "See https://example.com/page for details"}'
        cleaned = app.clean_json_response(raw)
        parsed = json.loads(cleaned)
        assert "https://example.com/page" in parsed["justification"]

    def test_double_slash_inside_string_not_stripped(self):
        raw = '{"path": "C:\\\\Users//file"}'
        cleaned = app.clean_json_response(raw)
        # The // inside the string should remain
        assert "//" in cleaned

    def test_real_comment_outside_string_still_stripped(self):
        raw = '{"key": "value"} // this is a real comment\n{"k2": "v2"}'
        cleaned = app.clean_json_response(raw)
        assert "// this is" not in cleaned
        assert '"key": "value"' in cleaned


# ─────────────────────────────────────────────────────────────────────────────
# 11. Patch: get_applicable_rules does not mutate defaults
# ─────────────────────────────────────────────────────────────────────────────
class TestRuleEngineDeepCopy:
    def test_defaults_not_mutated_by_extracted_rules(self):
        engine = app.InsuranceRuleEngine()
        original_pre_existing = engine.default_rules["waiting_periods"]["pre_existing"]

        # Simulate extracting a different waiting period
        engine.extracted_rules = {"waiting_periods": {"pre_existing": 36}}
        rules = engine.get_applicable_rules()
        assert rules["waiting_periods"]["pre_existing"] == 36

        # Defaults must remain unchanged
        assert engine.default_rules["waiting_periods"]["pre_existing"] == original_pre_existing

    def test_multiple_calls_do_not_accumulate(self):
        engine = app.InsuranceRuleEngine()
        engine.extracted_rules = {"waiting_periods": {"custom_rule": 6}}
        r1 = engine.get_applicable_rules()
        assert "custom_rule" in r1["waiting_periods"]

        engine.extracted_rules = {}
        r2 = engine.get_applicable_rules()
        assert "custom_rule" not in r2["waiting_periods"]


# ─────────────────────────────────────────────────────────────────────────────
# 12. Patch: QueryParser duration patterns & completeness
# ─────────────────────────────────────────────────────────────────────────────
class TestQueryParserDurationPatches:
    def setup_method(self):
        self.qp = app.QueryParser()

    def test_policy_purchased_n_months_ago(self):
        result = self.qp.parse_query("Maternity delivery, policy purchased 10 months ago")
        assert result["policy_duration"] == 10
        assert result["policy_duration_unit"] == "month"

    def test_policy_bought_n_years_ago(self):
        result = self.qp.parse_query("policy bought 2 years ago, cardiac surgery")
        assert result["policy_duration"] == 2
        assert result["policy_duration_unit"] == "year"

    def test_policy_n_years_old_without_is(self):
        """'policy 2 years old' (no 'is') should be parsed."""
        result = self.qp.parse_query("30-year-old female, policy 2 years old")
        assert result["policy_duration"] == 2
        assert result["age"] == 30

    def test_completeness_includes_policy_duration(self):
        result = self.qp.parse_query("3-month policy")
        assert result["policy_duration"] == 3
        assert result["completeness_score"] >= 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 13. Patch: confidence clamping
# ─────────────────────────────────────────────────────────────────────────────
class TestConfidenceClamping:
    @staticmethod
    def _failing_parser():
        p = MagicMock()
        p.parse.side_effect = Exception("forced fallback")
        return p

    def test_confidence_above_1_is_clamped(self):
        raw = json.dumps({
            "decision": "APPROVED", "amount": 10000,
            "justification": "page 1", "confidence": 1.5,
            "reasoning_steps": [], "source_pages": [1], "rule_violations": [],
        })
        result, _ = app.process_enhanced_response(
            raw, self._failing_parser(), {"violations": [], "confidence_impact": 0.0}
        )
        assert result["confidence"] <= 1.0

    def test_confidence_below_0_is_clamped(self):
        raw = json.dumps({
            "decision": "REJECTED", "amount": 0,
            "justification": "page 1", "confidence": -0.5,
            "reasoning_steps": [], "source_pages": [1], "rule_violations": [],
        })
        result, _ = app.process_enhanced_response(
            raw, self._failing_parser(), {"violations": [], "confidence_impact": 0.0}
        )
        assert result["confidence"] >= 0.0
