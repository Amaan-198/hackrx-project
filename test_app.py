import pytest
from unittest.mock import patch, Mock

# Mock the expensive imports before they are imported in app.py
with patch('langchain_community.embeddings.HuggingFaceEmbeddings') as mock_hf_embeddings, \
     patch('langchain_community.llms.Ollama') as mock_ollama, \
     patch('spacy.load') as mock_spacy_load:

    mock_hf_embeddings.return_value = Mock()
    mock_ollama.return_value = Mock()
    mock_spacy_load.return_value = None

    from app import QueryParser, InsuranceRuleEngine, ConfidenceCalculator

    @pytest.fixture
    def query_parser():
        return QueryParser()

    def test_parse_query_age_gender(query_parser):
        query = "46-year-old male, knee surgery"
        parsed = query_parser.parse_query(query)
        assert parsed["age"] == 46
        assert parsed["gender"] == "male"

    def test_parse_query_condition(query_parser):
        query = "claim for diabetes treatment"
        parsed = query_parser.parse_query(query)
        assert parsed["condition"] == "diabetes, treatment"
        assert parsed["treatment_type"] == "diabetes"

    def test_parse_query_policy_duration(query_parser):
        query = "3-month-old policy"
        parsed = query_parser.parse_query(query)
        assert parsed["policy_duration"] == 3
        assert parsed["policy_duration_unit"] == "month"

    def test_parse_query_amount(query_parser):
        query = "claim for rs. 50,000"
        parsed = query_parser.parse_query(query)
        assert parsed["amount_mentioned"] == 50000

    def test_get_missing_fields(query_parser):
        parsed = {"age": 46, "gender": "male", "condition": "knee surgery", "policy_duration": None}
        missing = query_parser.get_missing_fields(parsed)
        assert "policy duration" in missing

    def test_completeness_score(query_parser):
        query = "46-year-old male, knee surgery"
        parsed = query_parser.parse_query(query)
        assert parsed["completeness_score"] == 0.25 + 0.15 + 0.25

    @pytest.fixture
    def rule_engine():
        return InsuranceRuleEngine()

    def test_exclusion_extraction(rule_engine):
        policy_text = """
        Exclusions:
        • Cosmetic surgery, unless medically necessary
        • Experimental treatments
        • Self-inflicted injuries
        """
        rules = rule_engine.extract_rules_from_policy(policy_text)
        assert "exclusions" in rules
        assert "cosmetic surgery, unless medically necessary" in rules["exclusions"]
        assert "experimental treatments" in rules["exclusions"]
        assert "self-inflicted injuries" in rules["exclusions"]

    def test_validate_claim_age_too_low(rule_engine):
        parsed_query = {"age": 17, "condition": "something", "policy_duration": 12}
        validation = rule_engine.validate_claim(parsed_query)
        assert not validation["passed"]
        assert "below minimum entry age" in validation["violations"][0]

    def test_validate_claim_waiting_period(rule_engine):
        parsed_query = {"age": 30, "condition": "diabetes", "policy_duration": 12, "policy_duration_unit": "month"}
        validation = rule_engine.validate_claim(parsed_query)
        assert not validation["passed"]
        assert "Pre-existing condition" in validation["violations"][0]

    @pytest.fixture
    def confidence_calculator():
        return ConfidenceCalculator()

    def test_calculate_confidence(confidence_calculator):
        query = "test query"
        parsed_query = {"completeness_score": 0.8}
        retrieved_docs = [Mock(page_content="test content")]
        rule_validation = {"confidence_impact": 0.1}
        llm_response = {"confidence": 0.9, "decision": "APPROVED", "amount": 1000, "justification": "some justification"}
        confidence = confidence_calculator.calculate_comprehensive_confidence(query, parsed_query, retrieved_docs, rule_validation, llm_response)
        assert "final_confidence" in confidence
        assert confidence["final_confidence"] > 0.5
