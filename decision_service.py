import hashlib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from decision_core import (
    ClaimDecisionEngine,
    ConfidenceCalculator,
    InsuranceRuleEngine,
    QueryParser,
    build_policy_radar,
    create_enhanced_qa_chain,
    create_enhanced_vector_store,
    create_query_parser_llm,
    format_currency,
    normalize_whitespace,
)

load_dotenv()

API_UPLOAD_DIR = Path("temp") / "api_uploads"


class DecisionServiceError(RuntimeError):
    """Raised when the backend cannot prepare or evaluate a claim."""


@dataclass
class DecisionContext:
    file_hash: str
    document_name: str
    document_profile: Dict[str, Any]
    extracted_rules: Dict[str, Any]
    policy_radar: Dict[str, Any]
    engine: ClaimDecisionEngine
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class DecisionService:
    def __init__(self, upload_dir: Path | str = API_UPLOAD_DIR):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._contexts: Dict[str, DecisionContext] = {}
        self._contexts_lock = threading.Lock()

    def _sanitize_filename(self, filename: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename or "policy.pdf").name)
        if not cleaned.lower().endswith(".pdf"):
            cleaned = f"{cleaned}.pdf"
        return cleaned or "policy.pdf"

    def _write_policy_pdf(self, file_bytes: bytes, filename: str, file_hash: str) -> Path:
        safe_name = self._sanitize_filename(filename)
        target_path = self.upload_dir / f"{file_hash}_{safe_name}"
        if not target_path.exists():
            target_path.write_bytes(file_bytes)
        return target_path

    def _build_context(self, file_path: Path, file_hash: str) -> DecisionContext:
        vector_store, page_mapping, extracted_rules, document_profile = create_enhanced_vector_store(
            str(file_path),
            file_hash,
        )
        qa_chain, output_parser = create_enhanced_qa_chain(vector_store, page_mapping)
        query_parser = QueryParser(create_query_parser_llm())
        rule_engine = InsuranceRuleEngine()
        rule_engine.extracted_rules = extracted_rules
        confidence_calc = ConfidenceCalculator()
        policy_radar = build_policy_radar(document_profile["policy_text"], extracted_rules)
        engine = ClaimDecisionEngine(
            qa_chain,
            output_parser,
            query_parser,
            rule_engine,
            confidence_calc,
            page_mapping,
        )
        return DecisionContext(
            file_hash=file_hash,
            document_name=document_profile.get("document_name", file_path.name),
            document_profile=document_profile,
            extracted_rules=extracted_rules,
            policy_radar=policy_radar,
            engine=engine,
        )

    def _get_or_create_context(self, file_bytes: bytes, filename: str) -> DecisionContext:
        if not file_bytes:
            raise ValueError("Uploaded policy PDF is empty.")

        file_hash = hashlib.md5(file_bytes).hexdigest()
        with self._contexts_lock:
            cached = self._contexts.get(file_hash)
        if cached:
            return cached

        file_path = self._write_policy_pdf(file_bytes, filename, file_hash)
        try:
            context = self._build_context(file_path, file_hash)
        except Exception as exc:
            raise DecisionServiceError(f"Could not load the policy PDF: {exc}") from exc

        with self._contexts_lock:
            existing = self._contexts.get(file_hash)
            if existing:
                return existing
            self._contexts[file_hash] = context
            return context

    def evaluate_claim(self, file_bytes: bytes, filename: str, query: str) -> Dict[str, Any]:
        normalized_query = normalize_whitespace(query)
        if not normalized_query:
            raise ValueError("Claim query is required.")

        context = self._get_or_create_context(file_bytes, filename)
        try:
            with context.lock:
                result = context.engine.process_single_query(normalized_query)
        except Exception as exc:
            raise DecisionServiceError(f"Claim evaluation failed: {exc}") from exc

        decision = str(result.get("decision", "REJECTED")).upper()
        if decision == "APPROVED":
            decision_label = "ACCEPT"
        elif decision == "REJECTED":
            decision_label = "REJECT"
        else:
            decision_label = decision

        amount = result.get("amount", 0)
        return {
            "decision": decision,
            "decisionLabel": decision_label,
            "query": normalized_query,
            "documentName": context.document_name,
            "justification": result.get("justification", "No justification provided"),
            "amount": amount,
            "amountDisplay": format_currency(amount),
            "sourcePages": result.get("source_pages", []),
            "confidence": result.get("final_confidence", result.get("confidence", 0.0)),
            "timestamp": result.get("timestamp"),
            "ruleViolations": result.get("rule_violations", []),
            "sourceEvidence": result.get("source_evidence", []),
            "missingFields": result.get("missing_fields", []),
            "policyRadar": context.policy_radar,
        }


decision_service = DecisionService()
