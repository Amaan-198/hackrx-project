import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
AMBIGUOUS_POLICY_TERMS = [
    "reasonable and customary",
    "medically necessary",
    "subject to insurer approval",
    "as decided by the company",
    "customary charges",
    "standard charges",
    "clinically necessary",
    "non-medical expenses",
    "investigations as required",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


class QueryParser:
    def __init__(self, llm=None):
        self.llm = llm

    def parse_query(self, query: str) -> Dict[str, Any]:
        parsed = {
            "age": None,
            "gender": None,
            "condition": None,
            "location": None,
            "policy_duration": None,
            "policy_duration_unit": None,
            "treatment_type": None,
            "amount_mentioned": None,
            "completeness_score": 0.0,
        }
        query_lower = normalize_whitespace(query).lower()

        duration_patterns = [
            r"(\d+)[-\s]?(month|year)s?[-\s]?old\s*policy",
            r"policy\s*(?:is\s*)?(\d+)[-\s]?(month|year)s?\s*old",
            r"(\d+)[-\s]?(month|year)s?\s*policy",
            r"policy\s*for\s*(\d+)[-\s]?(month|year)s?",
            r"policy\s*(?:has\s+been\s+)?active\s+for\s*(\d+)[-\s]?(month|year)s?",
            r"policy\s+(?:purchased|bought|taken|started|active)\s+(\d+)\s*(month|year)s?\s*ago",
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if not match:
                continue
            parsed["policy_duration"] = int(match.group(1))
            parsed["policy_duration_unit"] = match.group(2).lower()
            query_lower = query_lower.replace(match.group(0), "", 1)
            break

        for pattern in [
            r"\b(\d+)[-\s]?year[-\s]?old\b",
            r"\b(\d+)[-\s]?y[.\-]?o\b",
            r"\bage\s*:?\s*(\d+)\b",
            r"\b(\d+)\s+years?\s+of\s+age\b",
        ]:
            match = re.search(pattern, query_lower)
            if not match:
                continue
            try:
                parsed["age"] = int(match.group(1))
            except (TypeError, ValueError):
                pass
            break

        if re.search(r"\bfemale\b|\bwoman\b|\blady\b|\bgirl\b", query_lower):
            parsed["gender"] = "female"
        elif re.search(r"\bmale\b|\bman\b|\bgentleman\b", query_lower):
            parsed["gender"] = "male"

        for city in ["mumbai", "delhi", "bangalore", "pune", "chennai", "kolkata", "hyderabad", "ahmedabad"]:
            if city in query_lower:
                parsed["location"] = city.title()
                break

        medical_keywords = [
            "surgery",
            "operation",
            "treatment",
            "therapy",
            "procedure",
            "diabetes",
            "heart",
            "cardiac",
            "knee",
            "hip",
            "cancer",
            "accident",
            "injury",
            "fracture",
            "maternity",
            "delivery",
            "hospitalization",
            "cataract",
        ]
        found_conditions = dedupe_keep_order([keyword for keyword in medical_keywords if keyword in query_lower])
        if found_conditions:
            parsed["condition"] = ", ".join(found_conditions)
            parsed["treatment_type"] = found_conditions[0]

        for pattern in [r"(?:₹|rs\.?|inr)\s*(\d+(?:,\d+)*)", r"(\d+(?:,\d+)*)\s*rupees?"]:
            match = re.search(pattern, query_lower)
            if not match:
                continue
            try:
                parsed["amount_mentioned"] = int(match.group(1).replace(",", ""))
            except ValueError:
                pass
            break

        parsed["completeness_score"] = self.calculate_completeness(parsed)
        return self.enrich_with_llm(query, parsed)

    def calculate_completeness(self, parsed: Dict[str, Any]) -> float:
        score = 0.0
        if parsed.get("age"):
            score += 0.15
        if parsed.get("gender"):
            score += 0.15
        if parsed.get("condition"):
            score += 0.25
        if parsed.get("policy_duration"):
            score += 0.15
        if parsed.get("location"):
            score += 0.1
        if parsed.get("amount_mentioned"):
            score += 0.1
        return min(1.0, round(score, 2))

    def get_missing_fields(self, parsed: Dict[str, Any]) -> List[str]:
        missing = []
        if not parsed.get("age"):
            missing.append("age")
        if not parsed.get("gender"):
            missing.append("gender")
        if not parsed.get("condition"):
            missing.append("medical condition/treatment")
        if not parsed.get("policy_duration"):
            missing.append("policy duration")
        return missing

    def should_use_llm_enrichment(self, parsed: Dict[str, Any]) -> bool:
        return self.llm is not None and (
            parsed.get("completeness_score", 0.0) < 0.55 or len(self.get_missing_fields(parsed)) >= 2
        )

    def _parse_llm_extraction(self, payload: str) -> Dict[str, Any]:
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def enrich_with_llm(self, query: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if not self.should_use_llm_enrichment(parsed):
            return parsed

        prompt = f"""
Extract structured insurance claim facts from this text.
Return JSON only with keys:
age, gender, condition, location, policy_duration, policy_duration_unit, amount_mentioned

Rules:
- Use null if not present.
- gender must be "male", "female", or null.
- policy_duration_unit must be "month", "year", or null.
- amount_mentioned must be numeric if present.

Claim:
{query}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            extracted = self._parse_llm_extraction(content)
        except Exception:
            return parsed
        if not extracted:
            return parsed

        merged = dict(parsed)
        for field in [
            "age",
            "gender",
            "condition",
            "location",
            "policy_duration",
            "policy_duration_unit",
            "amount_mentioned",
        ]:
            if merged.get(field):
                continue
            value = extracted.get(field)
            if value in (None, "", []):
                continue
            merged[field] = value

        if merged.get("condition") and not merged.get("treatment_type"):
            merged["treatment_type"] = str(merged["condition"]).split(",")[0].strip()
        merged["completeness_score"] = self.calculate_completeness(merged)
        return merged


class InsuranceRuleEngine:
    def __init__(self):
        self.default_rules = {
            "waiting_periods": {
                "pre_existing": 24,
                "maternity": 9,
                "specific_diseases": 12,
                "general_surgery": 1,
            },
            "age_limits": {"entry_age": {"min": 18, "max": 65}, "renewal_age": {"max": 80}},
        }
        self.extracted_rules = {}

    def extract_rules_from_policy(self, policy_text: str) -> Dict[str, Any]:
        rules: Dict[str, Any] = {}
        policy_lower = policy_text.lower()
        waiting_patterns = {
            "pre_existing": [
                r"pre[-\s]?existing.*?(\d+)[-\s]?(day|month|year)s?",
                r"waiting[-\s]?period.*?pre[-\s]?existing.*?(\d+)[-\s]?(day|month|year)s?",
                r"(\d+)[-\s]?(day|month|year)s?.*?waiting.*?pre[-\s]?existing",
            ],
            "maternity": [
                r"maternity.*?(\d+)[-\s]?(day|month|year)s?",
                r"pregnancy.*?waiting.*?(\d+)[-\s]?(day|month|year)s?",
                r"(\d+)[-\s]?(day|month|year)s?.*?maternity",
            ],
            "general": [
                r"initial waiting period[:\s-]*?(\d+)[-\s]?(day|month|year)s?",
                r"waiting[-\s]?period.*?(\d+)[-\s]?(day|month|year)s?",
                r"(\d+)[-\s]?(day|month|year)s?.*?waiting",
            ],
        }
        for category, patterns in waiting_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, policy_lower)
                if not matches:
                    continue
                for match in matches:
                    try:
                        duration = int(match[0])
                        unit = match[1].lower()
                        months = duration * 12 if unit.startswith("year") else max(1, round(duration / 30)) if unit.startswith("day") else duration
                        rules.setdefault("waiting_periods", {})[category] = months
                        break
                    except (IndexError, TypeError, ValueError):
                        continue
                if category in rules.get("waiting_periods", {}):
                    break

        min_age, max_age = None, None
        for pattern in [
            r"entry[-\s]?age.*?(\d+)[-\s]?to[-\s]?(\d+)",
            r"minimum[-\s]?age.*?(\d+)",
            r"maximum[-\s]?age.*?(\d+)",
            r"age\s*limit.*?(\d+)\s*-\s*(\d+)",
            r"eligible\s*age.*?(\d+)\s*-\s*(\d+)",
        ]:
            matches = re.findall(pattern, policy_lower)
            if not matches:
                continue
            first = matches[0]
            if isinstance(first, tuple) and len(first) == 2:
                min_age, max_age = int(first[0]), int(first[1])
                break
            if "minimum" in pattern:
                min_age = int(first)
            if "maximum" in pattern:
                max_age = int(first)
        if min_age is not None or max_age is not None:
            rules["age_limits"] = {"entry_age": {}}
            if min_age is not None:
                rules["age_limits"]["entry_age"]["min"] = min_age
            if max_age is not None:
                rules["age_limits"]["entry_age"]["max"] = max_age

        exclusions = []
        for section in self._extract_exclusion_sections(policy_lower):
            exclusions.extend(self._split_exclusion_candidates(section))
        if exclusions:
            rules["exclusions"] = self._filter_exclusions(exclusions)

        self.extracted_rules = rules
        return rules

    def _extract_exclusion_sections(self, policy_lower: str) -> List[str]:
        sections = []
        section_patterns = [
            r"following is a partial list of the policy exclusions\.?\s*please refer to the\s+policy document for the complete list of exclusions:\s*exclusions\s*(.+?)(?=\(?note:|waiting\s+period|payout\s+basis|$)",
            r"\bexclusions\s+we will not pay for any expenses incurred by insured.*?(?=general conditions)",
            r"exclusions?:\s*([^.]+(?:,[^.]+)*)",
        ]
        for pattern in section_patterns:
            sections.extend(re.findall(pattern, policy_lower, re.MULTILINE | re.DOTALL))
        return [normalize_whitespace(section) for section in sections if normalize_whitespace(section)]

    def _split_exclusion_candidates(self, section: str) -> List[str]:
        normalized = section.replace("•", "\n").replace("\r", "\n")
        normalized = normalize_whitespace(normalized)
        numbered_blocks = re.findall(r"(?:^|\s)(\d+)\.\s*(.+?)(?=(?:\s\d+\.\s)|$)", normalized, re.DOTALL)
        if numbered_blocks:
            return [
                summarized
                for _, block in numbered_blocks
                if (summarized := self._summarize_exclusion_candidate(block))
            ]

        normalized = re.sub(r"\s*-\s+", "\n", normalized)
        parts = re.split(r"[\n;,]", normalized)
        candidates = []
        for part in parts:
            if summarized := self._summarize_exclusion_candidate(part):
                candidates.append(summarized)
        return candidates

    def _summarize_exclusion_candidate(self, candidate: str) -> str:
        text = normalize_whitespace(candidate)
        if not text:
            return ""
        text = re.sub(r"\(code-[^)]+\)", "", text, flags=re.IGNORECASE)

        keyword_patterns = [
            r"investigation\s*&?\s*evaluation",
            r"rest cure,\s*rehabilitation and respite care",
            r"surgical treatment of obesity",
            r"change[-\s]?of[-\s]?gender treatments?",
            r"cosmetic or plastic surgery",
            r"hazardous(?:\s+or)?\s+adventure sports",
            r"sterility and infertility",
            r"maternity expenses",
            r"breach of law",
            r"rest cure",
        ]
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return normalize_whitespace(match.group(0))

        text = re.sub(
            r"^(?:expenses?\s+(?:for|related to|incurred on)\s+|admission primarily for\s+|treatment for\s+)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"[:\-–].*$", "", text).strip(" .")
        words = text.split()
        if len(words) > 10:
            text = " ".join(words[:10]).strip(" .")
        return text

    def _filter_exclusions(self, exclusions: List[str]) -> List[str]:
        blocked_phrases = {
            "the policy",
            "following is a partial list of the policy exclusions",
            "exclusions",
            "hereon",
            "however",
            "of the following:",
            "from one insurer to",
            "another insurer",
            "specified in the policy contract",
            "policy document for the complete list of exclusions",
        }
        blocked_substrings = [
            "policy document",
            "refer to",
            "please see",
            "waiting period",
            "sum insured",
            "scope of cover",
            "cost sharing",
            "claim process",
            "eligibility",
            "renewal",
            "definitions",
            "grievance",
            "appearing under the policy hereby stand deleted",
        ]
        cleaned = []
        for exclusion in exclusions:
            candidate = normalize_whitespace(exclusion.strip(" :-"))
            candidate = re.sub(r"^\d+\s*[.)]?\s*", "", candidate)
            candidate = candidate.strip(" .")
            if len(candidate) < 8:
                continue
            if candidate in blocked_phrases:
                continue
            if re.fullmatch(r"[\d\s]+", candidate):
                continue
            if not re.search(r"[a-z]{4}", candidate):
                continue
            if any(blocked in candidate for blocked in blocked_substrings):
                continue
            if candidate in {"pre-existing diseases", "30-day waiting period", "specified disease/procedure waiting period"}:
                continue
            if candidate.startswith(("the policy", "following", "terms and conditions", "shall", "with the", "of the policy")):
                continue
            if len(candidate.split()) > 14:
                continue
            cleaned.append(candidate)
        return dedupe_keep_order(cleaned)

    def get_applicable_rules(self) -> Dict[str, Any]:
        final_rules = copy.deepcopy(self.default_rules)
        for category, values in self.extracted_rules.items():
            if category in final_rules and isinstance(values, dict):
                final_rules[category].update(values)
            else:
                final_rules[category] = values
        return final_rules

    def validate_claim(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        if parsed_query.get("age") and parsed_query["age"] < 5:
            return {
                "passed": True,
                "violations": [],
                "warnings": [],
                "applicable_rules": [],
                "confidence_impact": 0.0,
                "rules_source": "skipped_invalid_age",
            }
        rules = self.get_applicable_rules()
        validation = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "applicable_rules": [],
            "confidence_impact": 0.0,
            "rules_source": "policy_specific" if self.extracted_rules else "default",
        }

        age = parsed_query.get("age")
        if age:
            limits = rules.get("age_limits", {}).get("entry_age", {})
            min_age = limits.get("min", 18)
            max_age = limits.get("max", 65)
            if age < min_age:
                validation["violations"].append(f"Age {age} below minimum entry age ({min_age}) as per policy")
                validation["confidence_impact"] -= 0.3
                validation["passed"] = False
            elif age > max_age:
                validation["violations"].append(f"Age {age} above maximum entry age ({max_age}) as per policy")
                validation["confidence_impact"] -= 0.3
                validation["passed"] = False

        condition = (parsed_query.get("condition") or "").lower()
        if parsed_query.get("policy_duration") and condition:
            try:
                duration = int(parsed_query.get("policy_duration") or 0)
            except (TypeError, ValueError):
                duration = 0
            unit = (parsed_query.get("policy_duration_unit") or "month").lower()
            duration_months = duration * 12 if unit.startswith("year") else duration
            waits = rules.get("waiting_periods", {})
            if any(key in condition for key in ["diabetes", "hypertension", "heart", "cardiac", "blood pressure"]):
                required_wait = waits.get("pre_existing", 24)
                if duration_months < required_wait:
                    validation["violations"].append(
                        f"Pre-existing condition ({condition}) requires {required_wait} months waiting period as per policy. Policy only {duration_months} months old."
                    )
                    validation["confidence_impact"] -= 0.4
                    validation["passed"] = False
                else:
                    validation["applicable_rules"].append(
                        f"Pre-existing condition waiting period ({required_wait} months) satisfied"
                    )
                    validation["confidence_impact"] += 0.1
            if any(key in condition for key in ["maternity", "delivery", "pregnancy"]):
                required_wait = waits.get("maternity", 9)
                if duration_months < required_wait:
                    validation["violations"].append(
                        f"Maternity claims require {required_wait} months waiting period as per policy. Policy only {duration_months} months old."
                    )
                    validation["confidence_impact"] -= 0.4
                    validation["passed"] = False

        exclusions = rules.get("exclusions", [])
        for exclusion in exclusions:
            if condition and re.search(r"\b" + re.escape(exclusion.lower()) + r"\b", condition):
                validation["violations"].append(f"Treatment '{condition}' contains excluded procedure: {exclusion}")
                validation["confidence_impact"] -= 0.5
                validation["passed"] = False
        return validation


class ConfidenceCalculator:
    def calculate_comprehensive_confidence(
        self,
        query: str,
        parsed_query: Dict[str, Any],
        retrieved_docs: List[Any],
        rule_validation: Dict[str, Any],
        llm_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        factors = {
            "query_completeness": parsed_query.get("completeness_score", 0.0),
            "doc_relevance": self.calculate_doc_relevance(query, retrieved_docs),
            "rule_impact": max(-0.5, min(0.3, rule_validation.get("confidence_impact", 0.0))),
            "llm_confidence": llm_response.get("confidence", 0.5),
            "response_consistency": self.calculate_response_consistency(llm_response),
        }
        weights = {
            "query_completeness": 0.2,
            "doc_relevance": 0.25,
            "rule_impact": 0.2,
            "llm_confidence": 0.25,
            "response_consistency": 0.1,
        }
        base = (
            factors["query_completeness"] * weights["query_completeness"]
            + factors["doc_relevance"] * weights["doc_relevance"]
            + factors["llm_confidence"] * weights["llm_confidence"]
            + factors["response_consistency"] * weights["response_consistency"]
        )
        final = max(0.0, min(1.0, base + (factors["rule_impact"] * weights["rule_impact"])))
        return {
            "final_confidence": final,
            "factor_breakdown": factors,
            "weights_used": weights,
            "explanation": self.generate_confidence_explanation(factors),
        }

    def calculate_doc_relevance(self, query: str, docs: List[Any]) -> float:
        if not docs:
            return 0.0
        query_words = {word for word in re.findall(r"[a-z0-9]+", query.lower()) if len(word) > 2}
        if not query_words:
            return 0.0
        total = 0.0
        for doc in docs:
            doc_text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            doc_words = {word for word in re.findall(r"[a-z0-9]+", doc_text.lower()) if len(word) > 2}
            total += len(query_words.intersection(doc_words)) / len(query_words)
        return min(1.0, total / len(docs))

    def calculate_response_consistency(self, response: Dict[str, Any]) -> float:
        decision = str(response.get("decision", "")).upper()
        amount = response.get("amount", 0)
        justification = normalize_whitespace(str(response.get("justification", "")))
        score = 1.0
        if decision == "APPROVED" and isinstance(amount, (int, float)) and amount == 0:
            score -= 0.3
        if decision == "REJECTED" and isinstance(amount, (int, float)) and amount != 0:
            score -= 0.4
        if len(justification) < 20:
            score -= 0.2
        return max(0.0, score)

    def generate_confidence_explanation(self, factors: Dict[str, float]) -> str:
        parts = []
        parts.append("Query is well specified" if factors["query_completeness"] > 0.7 else "Query is partially specified" if factors["query_completeness"] > 0.4 else "Query is missing important details")
        parts.append("Retrieved evidence is highly relevant" if factors["doc_relevance"] > 0.6 else "Retrieved evidence is moderately relevant" if factors["doc_relevance"] > 0.3 else "Retrieved evidence is weak")
        parts.append("Rule checks support the answer" if factors["rule_impact"] > 0.1 else "Rule checks are neutral" if factors["rule_impact"] > -0.1 else "Rule checks raise material concerns")
        return " | ".join(parts)


@st.cache_resource
def initialize_embedding_model():
    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=str(Path.home() / ".cache" / "huggingface"),
        )
        return embeddings, None
    except Exception as exc:
        return None, f"Error loading embedding model: {exc}"


@st.cache_resource
def create_enhanced_vector_store(file_path: str, file_hash: str = "") -> Tuple[FAISS, Dict[int, int], Dict[str, Any], Dict[str, Any]]:
    del file_hash
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Loading policy document...")
    progress_bar.progress(20)
    docs = PyPDFLoader(file_path).load()
    full_text = "\n".join(doc.page_content for doc in docs)
    status_text.text("Chunking policy pages...")
    progress_bar.progress(45)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    ).split_documents(docs)
    page_mapping: Dict[int, int] = {}
    for index, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page", 0) + 1
        page_mapping[index] = page_number
        chunk.metadata["page_number"] = page_number
        chunk.metadata["chunk_id"] = index
    status_text.text("Creating embeddings...")
    progress_bar.progress(70)
    embeddings, error = initialize_embedding_model()
    if error:
        raise RuntimeError(error)
    vector_store = FAISS.from_documents(chunks, embeddings)
    status_text.text("Extracting policy rules...")
    progress_bar.progress(90)
    extracted_rules = InsuranceRuleEngine().extract_rules_from_policy(full_text)
    progress_bar.progress(100)
    status_text.text(f"Processed {len(chunks)} chunks from {len(docs)} pages and extracted {len(extracted_rules)} rule groups.")
    return vector_store, page_mapping, extracted_rules, {
        "policy_text": full_text,
        "page_count": len(docs),
        "chunk_count": len(chunks),
        "document_name": Path(file_path).name,
    }


def create_enhanced_prompt_template() -> Tuple[PromptTemplate, StructuredOutputParser]:
    parser = StructuredOutputParser.from_response_schemas(
        [
            ResponseSchema(name="decision", description="APPROVED or REJECTED"),
            ResponseSchema(name="amount", description="Numeric payout if available. Use 0 if rejected."),
            ResponseSchema(name="justification", description="Specific policy support with page references"),
            ResponseSchema(name="confidence", description="Decimal score between 0.0 and 1.0"),
            ResponseSchema(name="reasoning_steps", description="Short step-by-step reasoning as a list"),
            ResponseSchema(name="source_pages", description="List of page numbers supporting the answer"),
            ResponseSchema(name="rule_violations", description="List of violated policy rules, if any"),
        ]
    )
    prompt = PromptTemplate(
        template="""
You are an expert insurance claims analyst.
Use only the supplied policy context.
Every justification must mention page numbers.
Return APPROVED only when the policy evidence clearly supports the claim.
If evidence is incomplete, the facts are missing, or the policy support is weak, choose REJECTED and explain what is missing.
Return valid JSON only.

Policy context:
{context}

Claim query:
{question}

{format_instructions}
""",
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt, parser


def create_enhanced_qa_chain(vs: FAISS, page_mapping: Dict):
    del page_mapping
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
    llm = ChatGroq(model=MODEL_NAME, temperature=0.1, groq_api_key=groq_api_key)
    prompt, parser = create_enhanced_prompt_template()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain, parser


@st.cache_resource
def create_query_parser_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return None
    return ChatGroq(model=MODEL_NAME, temperature=0.0, groq_api_key=groq_api_key)


def clean_json_response(response: str) -> str:
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        in_string = False
        escape_next = False
        cut_index = len(line)
        for index, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string and char == "/" and index + 1 < len(line) and line[index + 1] == "/":
                cut_index = index
                break
        cleaned_lines.append(line[:cut_index])
    response = "\n".join(cleaned_lines)
    response = re.sub(r"/\*.*?\*/", "", response, flags=re.DOTALL)
    response = re.sub(r",(\s*[}\]])", r"\1", response)
    return response.strip()


def _normalize_amount(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return 0
    cleaned = normalize_whitespace(str(value))
    candidate = cleaned.lower().replace("inr", "").replace("rs.", "").replace("rs", "").replace(",", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        return int(float(candidate))
    return cleaned


def _normalize_source_pages(value: Any) -> List[int]:
    if isinstance(value, str):
        return [int(match) for match in re.findall(r"page\s*(\d+)", value, re.IGNORECASE)]
    if isinstance(value, list):
        pages = []
        for item in value:
            if isinstance(item, int):
                pages.append(item)
            elif isinstance(item, float):
                pages.append(int(item))
            elif isinstance(item, str):
                match = re.search(r"page\s*(\d+)", item, re.IGNORECASE) or re.search(r"\d+", item)
                if match:
                    pages.append(int(match.group(1) if match.lastindex else match.group(0)))
        return pages
    return []


def _normalize_reasoning_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]
    if isinstance(value, str):
        parts = re.split(r"(?:\r?\n)+|\s+\d+\.\s+", value)
        return [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]
    return []


def process_enhanced_response(response: str, parser, rule_validation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    try:
        cleaned_response = clean_json_response(response)
        parse_method = "structured"
        try:
            parsed = parser.parse(cleaned_response)
        except Exception as parser_error:
            try:
                parsed = json.loads(cleaned_response)
                parse_method = "direct_json"
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
                if not match:
                    raise ValueError("No JSON structure found in response") from parser_error
                parsed = json.loads(match.group())
                parse_method = "json_extraction"
        if not isinstance(parsed, dict):
            raise ValueError("Parsed response is not a JSON object")

        defaults = {
            "decision": "REJECTED",
            "amount": 0,
            "justification": "No justification provided",
            "confidence": 0.5,
            "reasoning_steps": [],
            "source_pages": [],
            "rule_violations": [],
        }
        for field, default in defaults.items():
            parsed.setdefault(field, default)

        decision = normalize_whitespace(str(parsed.get("decision", ""))).upper().replace(" ", "_")
        parsed["decision"] = {
            "APPROVE": "APPROVED",
            "REJECT": "REJECTED",
            "NEEDS_CLARIFICATION": "REJECTED",
            "NEED_CLARIFICATION": "REJECTED",
            "CLARIFICATION_REQUIRED": "REJECTED",
            "REQUIRES_CLARIFICATION": "REJECTED",
        }.get(decision, decision)
        if parsed["decision"] not in {"APPROVED", "REJECTED"}:
            parsed["decision"] = "REJECTED"

        parsed["amount"] = _normalize_amount(parsed.get("amount"))
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.5
        parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
        parsed["source_pages"] = _normalize_source_pages(parsed.get("source_pages"))
        parsed["reasoning_steps"] = _normalize_reasoning_steps(parsed.get("reasoning_steps"))

        raw_violations = parsed.get("rule_violations", [])
        if isinstance(raw_violations, str):
            violations = [normalize_whitespace(raw_violations)] if raw_violations.strip() else []
        elif isinstance(raw_violations, list):
            violations = [normalize_whitespace(str(item)) for item in raw_violations if normalize_whitespace(str(item))]
        else:
            violations = []
        violations = [
            item
            for item in violations
            if item.lower() not in {"none", "null", "n/a", "na", "no rule violations"}
        ]
        violations = dedupe_keep_order(violations + list(rule_validation.get("violations", [])))
        parsed["rule_violations"] = violations

        justification = normalize_whitespace(str(parsed.get("justification", ""))) or "No justification provided"
        if decision in {"NEEDS_CLARIFICATION", "NEED_CLARIFICATION", "CLARIFICATION_REQUIRED", "REQUIRES_CLARIFICATION"}:
            prefix = "Rejected because the claim details or policy evidence were not strong enough to approve."
            if prefix not in justification:
                justification = f"{prefix} {justification}".strip()
        if violations and parsed["decision"] == "APPROVED":
            parsed["decision"] = "REJECTED"
            parsed["amount"] = 0
            prefix = f"Rule validation flagged: {'; '.join(violations)}."
            if prefix not in justification:
                justification = f"{prefix} {justification}".strip()
        if parsed["decision"] == "REJECTED":
            parsed["amount"] = 0
        parsed["justification"] = justification
        return parsed, parse_method
    except Exception as exc:
        return {
            "error": f"Parsing failed: {exc}",
            "raw_response": response,
            "decision": "ERROR",
            "amount": 0,
            "justification": "System error occurred",
            "confidence": 0.0,
            "reasoning_steps": ["System error during processing"],
            "source_pages": [],
            "rule_violations": [],
        }, "error"


def format_currency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"Rs {int(value):,}" if value else "Rs 0"
    cleaned = normalize_whitespace(str(value))
    return cleaned or "Not specified"


def extract_source_evidence(retrieved_docs: List[Any], limit: int = 4) -> List[Dict[str, Any]]:
    evidence = []
    seen = set()
    for doc in retrieved_docs:
        page_number = doc.metadata.get("page_number")
        if page_number is None:
            page_number = doc.metadata.get("page", 0) + 1
        snippet = normalize_whitespace(getattr(doc, "page_content", ""))
        if not snippet:
            continue
        snippet = snippet[:260] + ("..." if len(snippet) > 260 else "")
        fingerprint = (page_number, snippet)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        evidence.append({"page": page_number, "snippet": snippet})
        if len(evidence) >= limit:
            break
    return evidence


def build_claim_query(
    raw_description: str = "",
    age: int = 0,
    gender: str = "Not specified",
    treatment: str = "",
    policy_duration: int = 0,
    policy_duration_unit: str = "month",
    location: str = "Unspecified",
    amount: int = 0,
) -> str:
    parts = []
    if age:
        parts.append(f"{age}-year-old")
    if gender and gender != "Not specified":
        parts.append(gender.lower())
    if treatment:
        parts.append(treatment.lower())
    if location and location != "Unspecified":
        parts.append(f"in {location}")
    if policy_duration:
        parts.append(f"policy {policy_duration} {policy_duration_unit}s old")
    if amount:
        parts.append(f"claim amount Rs {amount}")
    structured = ", ".join(parts)
    raw_description = normalize_whitespace(raw_description)
    if raw_description and structured:
        return f"{raw_description}. Structured claim facts: {structured}."
    return raw_description or structured


def _collect_waiting_period_mentions(policy_text: str, keywords: List[str]) -> List[int]:
    values = set()
    lowered = policy_text.lower()
    for keyword in keywords:
        for match in re.finditer(keyword, lowered):
            window = lowered[max(0, match.start() - 40): match.end() + 90]
            for duration, unit in re.findall(r"(\d+)[-\s]?(month|year)s?", window):
                values.add(int(duration) * 12 if unit.startswith("year") else int(duration))
    return sorted(values)


def build_policy_radar(policy_text: str, extracted_rules: Dict[str, Any]) -> Dict[str, Any]:
    lowered = normalize_whitespace(policy_text).lower()
    ambiguity_flags = [f"Uses '{term}' language" for term in AMBIGUOUS_POLICY_TERMS if term in lowered]
    contradictions = []
    for label, mentions in [
        ("Pre-existing", _collect_waiting_period_mentions(lowered, ["pre-existing", "pre existing"])),
        ("Maternity", _collect_waiting_period_mentions(lowered, ["maternity", "pregnancy"])),
        ("General", _collect_waiting_period_mentions(lowered, ["waiting period"])),
    ]:
        if len(mentions) > 1:
            contradictions.append(f"{label} waiting period appears with multiple values: {', '.join(str(v) for v in mentions)} months.")
    waits = extracted_rules.get("waiting_periods", {}) if extracted_rules else {}
    missing_signals = []
    if "pre_existing" not in waits:
        missing_signals.append("No clear pre-existing waiting period extracted")
    if "maternity" not in waits:
        missing_signals.append("No clear maternity waiting period extracted")
    if not extracted_rules.get("age_limits"):
        missing_signals.append("No explicit age limit extracted")
    if not extracted_rules.get("exclusions"):
        missing_signals.append("No exclusions list extracted")
    highlights = [f"{name.replace('_', ' ').title()}: {months} months" for name, months in waits.items()]
    if extracted_rules.get("age_limits", {}).get("entry_age"):
        limits = extracted_rules["age_limits"]["entry_age"]
        highlights.append(f"Entry age window: {limits.get('min', '?')} to {limits.get('max', '?')}")
    if extracted_rules.get("exclusions"):
        highlights.append(f"Extracted exclusions: {len(extracted_rules['exclusions'])}")
    return {
        "ambiguity_flags": ambiguity_flags,
        "contradictions": contradictions,
        "missing_signals": missing_signals,
        "highlights": highlights,
        "ambiguity_count": len(ambiguity_flags),
        "contradiction_count": len(contradictions),
        "missing_count": len(missing_signals),
    }


def build_policy_brief(extracted_rules: Dict[str, Any], document_profile: Dict[str, Any], policy_radar: Dict[str, Any]) -> Dict[str, Any]:
    exclusions = extracted_rules.get("exclusions", [])
    return {
        "metrics": [
            ("Pages", str(document_profile.get("page_count", 0))),
            ("Chunks", str(document_profile.get("chunk_count", 0))),
            ("Rule groups", str(len(extracted_rules))),
            ("Radar alerts", str(policy_radar.get("ambiguity_count", 0) + policy_radar.get("contradiction_count", 0))),
        ],
        "rule_chips": [
            f"Pre-existing {extracted_rules.get('waiting_periods', {}).get('pre_existing')}m" if extracted_rules.get("waiting_periods", {}).get("pre_existing") else "",
            f"Maternity {extracted_rules.get('waiting_periods', {}).get('maternity')}m" if extracted_rules.get("waiting_periods", {}).get("maternity") else "",
            f"{len(exclusions)} exclusions" if exclusions else "No exclusions extracted",
        ],
    }


def create_appeal_draft(result: Dict[str, Any], policy_radar: Dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in result.get("source_pages", [])) or "the cited policy pages"
    violations = "; ".join(result.get("rule_violations", [])) or "the stated denial reason"
    contradiction_note = ""
    if policy_radar.get("contradictions"):
        contradiction_note = f" The policy radar also flagged a wording conflict that should be manually reviewed: {policy_radar['contradictions'][0]}"
    return (
        "Subject: Request for detailed reconsideration of claim decision\n\n"
        "Hello,\n\n"
        "I am requesting a detailed review of the current claim assessment. "
        f"The present outcome appears to rely on {violations}. "
        f"The supporting policy references currently point to pages {pages}.{contradiction_note}\n\n"
        "Please confirm the exact clause relied upon, the interpretation applied, and whether any rider or exception changes the outcome. "
        "I am ready to provide discharge summary, treatment records, policy schedule, and billing documents for a complete review.\n\n"
        "Regards"
    )


def build_action_plan(result: Dict[str, Any], policy_radar: Dict[str, Any]) -> Dict[str, Any]:
    decision = result.get("decision", "REJECTED")
    if decision == "APPROVED":
        return {
            "headline": "Presentation-ready approval path",
            "items": [
                "Lead with the cited pages so the decision feels policy-backed, not model-backed.",
                "Use the scenario lab to prove the system reacts predictably when claim facts change.",
                "Mention confidence only after the coverage story is clear.",
            ],
            "draft": "This case is strong because the answer is policy-backed, not just model-generated, and the what-if lab shows the outcome moving when facts change.",
        }
    if decision == "REJECTED":
        items = [
            "Show the exact rule violation first so the rejection feels grounded.",
            "Use the what-if lab to prove the outcome flips when waiting period or age assumptions change.",
            "If the radar flags contradictions, mention that human review is still needed for disputed wording.",
        ]
        if policy_radar.get("contradictions"):
            items.append("Point to the policy contradiction flag as a reason to escalate the case.")
        return {"headline": "Appeal and escalation path", "items": items, "draft": create_appeal_draft(result, policy_radar)}
    return {
        "headline": "Conservative rejection path",
        "items": [
            "Call out the missing or weak evidence before discussing payout.",
            "Show the cited pages so the rejection still feels policy-backed.",
            "Use a stronger, more complete query if you want to demonstrate the accept path next.",
        ],
        "draft": "The system stays conservative when the claim facts or policy support are not strong enough to approve, so the demo never drifts into fake certainty.",
    }


def build_counterfactual_queries(original_query: str, parsed_query: Dict[str, Any], applicable_rules: Dict[str, Any]) -> List[Dict[str, str]]:
    waiting_periods = applicable_rules.get("waiting_periods", {})
    condition = (parsed_query.get("condition") or "").lower()
    recommended_wait = waiting_periods.get("general", 1)
    if any(keyword in condition for keyword in ["diabetes", "heart", "cardiac", "hypertension"]):
        recommended_wait = waiting_periods.get("pre_existing", 24)
    if any(keyword in condition for keyword in ["maternity", "delivery", "pregnancy"]):
        recommended_wait = waiting_periods.get("maternity", 9)
    age_limit = applicable_rules.get("age_limits", {}).get("entry_age", {}).get("max", 65)
    scenarios = [
        {"title": "If the policy were older", "caption": "Shows how the answer changes once waiting periods are satisfied.", "query": f"{original_query}. Assume the policy has been active for {max(recommended_wait + 2, 12)} months."},
        {"title": "If the policy were brand new", "caption": "Stress-tests early-tenure claims where waiting periods matter most.", "query": f"{original_query}. Assume the policy is only 1 month old."},
        {"title": "If the patient were older", "caption": "Checks whether age limits can break the same claim pattern.", "query": f"{original_query}. Assume the patient is {max(age_limit + 5, 70)} years old."},
    ]
    results = []
    seen = {normalize_whitespace(original_query)}
    for scenario in scenarios:
        query = normalize_whitespace(scenario["query"])
        if query in seen:
            continue
        seen.add(query)
        scenario["query"] = query
        results.append(scenario)
    return results


class ClaimDecisionEngine:
    def __init__(self, qa_chain, parser, query_parser, rule_engine, confidence_calc, page_mapping):
        self.qa_chain = qa_chain
        self.parser = parser
        self.query_parser = query_parser
        self.rule_engine = rule_engine
        self.confidence_calc = confidence_calc
        self.page_mapping = page_mapping

    def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        results = []
        for index, query in enumerate(queries):
            try:
                result = self.process_single_query(query)
                result["batch_index"] = index
                result["query"] = query
                results.append(result)
            except Exception as exc:
                results.append(
                    {
                        "batch_index": index,
                        "query": query,
                        "error": str(exc),
                        "decision": "ERROR",
                        "amount": 0,
                        "confidence": 0.0,
                        "final_confidence": 0.0,
                        "rule_violations": [],
                        "source_pages": [],
                        "reasoning_steps": [],
                        "justification": "Processing error occurred",
                        "parsed_query": {"completeness_score": 0},
                        "rule_validation": {"violations": [], "confidence_impact": 0.0},
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        return results

    def process_single_query(self, query: str) -> Dict[str, Any]:
        query = normalize_whitespace(query)
        if not query:
            raise ValueError("Claim query is empty.")
        parsed_query = self.query_parser.parse_query(query)
        rule_validation = self.rule_engine.validate_claim(parsed_query)
        llm_result = self.qa_chain.invoke({"query": query})
        processed_response, parse_method = process_enhanced_response(
            llm_result.get("result", ""),
            self.parser,
            rule_validation,
        )
        retrieved_docs = llm_result.get("source_documents", [])
        source_evidence = extract_source_evidence(retrieved_docs)
        if "error" not in processed_response:
            confidence = self.confidence_calc.calculate_comprehensive_confidence(
                query,
                parsed_query,
                retrieved_docs,
                rule_validation,
                processed_response,
            )
            processed_response["final_confidence"] = confidence["final_confidence"]
            processed_response["confidence_breakdown"] = confidence["factor_breakdown"]
            processed_response["confidence_explanation"] = confidence["explanation"]
        else:
            processed_response.setdefault("final_confidence", 0.0)
        if not processed_response.get("source_pages"):
            processed_response["source_pages"] = [item["page"] for item in source_evidence]
        processed_response.update(
            {
                "parsed_query": parsed_query,
                "rule_validation": rule_validation,
                "parse_method": parse_method,
                "timestamp": datetime.now().isoformat(),
                "missing_fields": self.query_parser.get_missing_fields(parsed_query),
                "source_evidence": source_evidence,
                "query": query,
            }
        )
        return processed_response


BatchProcessor = ClaimDecisionEngine


def run_scenario_lab(engine: ClaimDecisionEngine, base_query: str, base_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenario_results = []
    for scenario in build_counterfactual_queries(
        base_query,
        base_result.get("parsed_query", {}),
        engine.rule_engine.get_applicable_rules(),
    ):
        try:
            result = engine.process_single_query(scenario["query"])
        except Exception as exc:
            result = {"decision": "ERROR", "amount": 0, "justification": str(exc)}
        result["amount_display"] = format_currency(result.get("amount", 0))
        scenario_results.append({"title": scenario["title"], "caption": scenario["caption"], "result": result})
    return scenario_results
