# Enhanced Decision Co-Pilot with Complete Feature Set
import streamlit as st
import os
import json
import re
import spacy
from typing import Dict, Any, Optional, List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.documents import Document
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import hashlib

# --- Configuration ---
TEMP_DIR = "temp"
MODEL_NAME = "llama3:8b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Enhanced UI Configuration ---
st.set_page_config(
    page_title="Decision Co-Pilot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .confidence-high { background-color: #d4edda; padding: 0.5rem; border-radius: 0.5rem; }
    .confidence-medium { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.5rem; }
    .confidence-low { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.5rem; }
    .reasoning-box { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #007bff; }
    .stSpinner > div { text-align: center; }
    .stDownloadButton>button { width: 100%; }
    .json-box { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# --- Smart Query Preprocessing ---
class QueryParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query"""
        parsed = {
            "age": None,
            "gender": None,
            "condition": None,
            "location": None,
            "policy_duration": None,
            "policy_duration_unit": None,
            "treatment_type": None,
            "amount_mentioned": None,
            "completeness_score": 0.0
        }
        
        query_lower = query.lower()
        
        # Extract age
        
        # Avoid matching "3 years old policy" as patient age
        duration_patterns = [
            r'(\d+)[-\s]?(month|year)s?[-\s]?old\s*policy',
            r'policy\s*is\s*(\d+)[-\s]?(month|year)s?\s*old',
            r'(\d+)[-\s]?(month|year)s?\s*policy',
            r'policy\s*for\s*(\d+)[-\s]?(month|year)s?'
        ]


        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Extract and REMOVE this match from query
                parsed["policy_duration"] = int(match.group(1))
                parsed["policy_duration_unit"] = match.group(2).lower() if match.group(2) else "month"
                # Remove matched string to prevent false age detection
                query_lower = query_lower.replace(match.group(0), "", 1)
                break
        age_patterns = [
            r'\b(\d+)[-\s]?year[-\s]?old\b',
            r'\b(\d+)[-\s]?y[.\-]?o\b',
            r'\bage\s*:?\s*(\d+)\b',
            r'\b(\d+)\s+years?\s+of\s+age\b'
            ]
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    parsed["age"] = int(match.group(1))
                    parsed["completeness_score"] += 0.25
                except (ValueError, TypeError):
                    pass
                break
        # Extract gender
        if any(word in query_lower for word in ["male", "man", "m", "gentleman"]):
            parsed["gender"] = "male"
            parsed["completeness_score"] += 0.15
        elif any(word in query_lower for word in ["female", "woman", "f", "lady"]):
            parsed["gender"] = "female"
            parsed["completeness_score"] += 0.15
        
        # Extract location
        indian_cities = ["mumbai", "delhi", "bangalore", "pune", "chennai", "kolkata", "hyderabad", "ahmedabad"]
        for city in indian_cities:
            if city in query_lower:
                parsed["location"] = city.title()
                parsed["completeness_score"] += 0.1
                break
        
        # Extract medical conditions/treatments
        medical_keywords = [
            "surgery", "operation", "treatment", "therapy", "procedure",
            "diabetes", "heart", "cardiac", "knee", "hip", "cancer",
            "accident", "injury", "fracture", "maternity", "delivery"
        ]
        
        found_conditions = []
        for keyword in medical_keywords:
            if keyword in query_lower:
                found_conditions.append(keyword)
        
        if found_conditions:
            found_conditions.sort()
            parsed["condition"] = ", ".join(found_conditions)
            parsed["treatment_type"] = found_conditions[0]
            parsed["completeness_score"] += 0.25
        
        # Extract amount if mentioned
        amount_patterns = [r'₹\s*(\d+(?:,\d+)*)', r'rs\.?\s*(\d+(?:,\d+)*)', r'(\d+(?:,\d+)*)\s*rupees?']
        for pattern in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    parsed["amount_mentioned"] = int(amount_str)
                    parsed["completeness_score"] += 0.1
                except ValueError:
                    pass
                break
        
        return parsed
    
    def get_missing_fields(self, parsed: Dict) -> List[str]:
        """Identify missing critical information"""
        missing = []
        if not parsed["age"]:
            missing.append("age")
        if not parsed["gender"]:
            missing.append("gender")
        if not parsed["condition"]:
            missing.append("medical condition/treatment")
        if not parsed["policy_duration"]:
            missing.append("policy duration")
        return missing

# --- Rule-Based Validation Engine ---
class InsuranceRuleEngine:
    def __init__(self):
        self.default_rules = {
            "waiting_periods": {
                "pre_existing": 24,
                "maternity": 9,
                "specific_diseases": 12,
                "general_surgery": 1
            },
            "age_limits": {
                "entry_age": {"min": 18, "max": 65},
                "renewal_age": {"max": 80}
            }
        }
        self.extracted_rules = {}
    
    def extract_rules_from_policy(self, policy_text: str) -> Dict:
        """Extract policy-specific rules from the document text"""
        rules = {}
        
        policy_lower = policy_text.lower()
        
        # Extract waiting periods using regex patterns
        waiting_patterns = {
            "pre_existing": [
                r"pre[-\s]?existing.*?(\d+)[-\s]?(month|year)s?",
                r"waiting[-\s]?period.*?pre[-\s]?existing.*?(\d+)[-\s]?(month|year)s?",
                r"(\d+)[-\s]?(month|year)s?.*?waiting.*?pre[-\s]?existing"
            ],
            "maternity": [
                r"maternity.*?(\d+)[-\s]?(month|year)s?",
                r"pregnancy.*?waiting.*?(\d+)[-\s]?(month|year)s?",
                r"(\d+)[-\s]?(month|year)s?.*?maternity"
            ],
            "general": [
                r"waiting[-\s]?period.*?(\d+)[-\s]?(month|year)s?",
                r"(\d+)[-\s]?(month|year)s?.*?waiting"
            ]
        }
        
        for condition, patterns in waiting_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, policy_lower)
                if matches:
                    for match in matches:
                        try:
                            duration = int(match[0])
                            unit = match[1].lower() if len(match) > 1 else "month"
                            months = duration * 12 if unit.startswith('year') else duration
                            
                            if "waiting_periods" not in rules:
                                rules["waiting_periods"] = {}
                            rules["waiting_periods"][condition] = months
                            break
                        except (ValueError, IndexError):
                            continue
                if condition in rules.get("waiting_periods", {}):
                    break
        
        # Extract age limits
        age_patterns = [
            r"entry[-\s]?age.*?(\d+)[-\s]?to[-\s]?(\d+)",
            r"minimum[-\s]?age.*?(\d+)",
            r"maximum[-\s]?age.*?(\d+)",
            r"age\s*limit.*?(\d+)\s*-\s*(\d+)",
            r"eligible\s*age.*?(\d+)\s*-\s*(\d+)"
        ]
        
        min_age, max_age = None, None
        for pattern in age_patterns:
            matches = re.findall(pattern, policy_lower)
            if matches:
                if len(matches[0]) == 2:  # min-max pattern
                    try:
                        min_age = int(matches[0][0])
                        max_age = int(matches[0][1])
                    except (ValueError, IndexError):
                        pass
                elif "minimum" in pattern and matches:
                    try:
                        min_age = int(matches[0])
                    except (ValueError, IndexError):
                        pass
                elif "maximum" in pattern and matches:
                    try:
                        max_age = int(matches[0])
                    except (ValueError, IndexError):
                        pass
        
        if min_age is not None or max_age is not None:
            rules["age_limits"] = {"entry_age": {}}
            if min_age is not None:
                rules["age_limits"]["entry_age"]["min"] = min_age
            if max_age is not None:
                rules["age_limits"]["entry_age"]["max"] = max_age
        
        # Extract exclusions
        exclusion_patterns = [
            r"exclusions?:\s*\n((?:\s*•[^\n]+\n)+)",
            r"not[-\s]?covered:\s*\n((?:\s*•[^\n]+\n)+)",
            r"excluded[-\s]?treatments?:\s*\n((?:\s*•[^\n]+\n)+)",
            r"conditions?\s*not\s*covered:\s*\n((?:\s*•[^\n]+\n)+)"
        ]
        
        exclusions = []
        for pattern in exclusion_patterns:
            matches = re.findall(pattern, policy_lower, re.MULTILINE)
            for match in matches:
                # Clean and split exclusions
                items = [item.strip().lstrip('•').strip() for item in match.strip().split('\n') if item.strip()]
                exclusions.extend(items)

        # Post-processing to remove noise
        if exclusions:
            processed_exclusions = []
            for ex in exclusions:
                # Remove short, meaningless exclusions
                if len(ex.split()) >= 2 and len(ex) > 15:
                    processed_exclusions.append(ex)
            rules["exclusions"] = processed_exclusions
        
        self.extracted_rules = rules
        return rules
    
    def get_applicable_rules(self) -> Dict:
        """Get rules with policy-specific overriding defaults"""
        final_rules = self.default_rules.copy()
        
        if self.extracted_rules:
            for category, values in self.extracted_rules.items():
                if category in final_rules:
                    if isinstance(values, dict):
                        final_rules[category].update(values)
                    else:
                        final_rules[category] = values
                else:
                    final_rules[category] = values
        
        return final_rules
    
    def validate_claim(self, parsed_query: Dict) -> Dict[str, Any]:
        """Perform rule-based validation using policy-specific rules"""
        if parsed_query.get("age") and parsed_query["age"] < 5:  # Impossible insurance age
        # Likely parsing error - skip validation
            return {
                "passed": True,
                "violations": [],
                "confidence_impact": 0.0
                }
        current_rules = self.get_applicable_rules()
        
        validation = {
            "passed": True,
            "violations": [],
            "warnings": [],
            "applicable_rules": [],
            "confidence_impact": 0.0,
            "rules_source": "policy_specific" if self.extracted_rules else "default"
        }
        
        # Age validation
        if parsed_query.get("age"):
            age = parsed_query["age"]
            age_limits = current_rules.get("age_limits", {}).get("entry_age", {})
            min_age = age_limits.get("min", 18)
            max_age = age_limits.get("max", 65)
            
            if age < min_age:
                validation["violations"].append(f"Age {age} below minimum entry age ({min_age}) as per policy")
                validation["passed"] = False
                validation["confidence_impact"] -= 0.3
            elif age > max_age:
                validation["violations"].append(f"Age {age} above maximum entry age ({max_age}) as per policy")
                validation["passed"] = False
                validation["confidence_impact"] -= 0.3
        
        # Waiting period validation
        if parsed_query.get("policy_duration") and parsed_query.get("condition"):
            policy_duration_raw = parsed_query.get("policy_duration")
            
            try:
                if isinstance(policy_duration_raw, str):
                    duration = int(policy_duration_raw)
                else:
                    duration = int(policy_duration_raw) if policy_duration_raw else 0
            except (ValueError, TypeError):
                duration = 0

            unit = (parsed_query.get("policy_duration_unit") or "month").lower()
            duration_months = duration * 12 if unit.startswith("year") else duration

            condition = parsed_query["condition"].lower()
            waiting_periods = current_rules.get("waiting_periods", {})
            
            # Check for pre-existing conditions
            pre_existing_keywords = ["diabetes", "hypertension", "heart", "cardiac", "blood pressure"]
            if any(keyword in condition for keyword in pre_existing_keywords):
                required_wait = waiting_periods.get("pre_existing", 24)
                if duration_months < required_wait:
                    validation["violations"].append(
                        f"Pre-existing condition ({condition}) requires {required_wait} months waiting period as per policy. Policy only {duration_months} months old."
                    )
                    validation["passed"] = False
                    validation["confidence_impact"] -= 0.4
                else:
                    validation["applicable_rules"].append(f"Pre-existing condition waiting period ({required_wait} months) satisfied")
                    validation["confidence_impact"] += 0.1
            
            # Check maternity
            if "maternity" in condition or "delivery" in condition or "pregnancy" in condition:
                required_wait = waiting_periods.get("maternity", 9)
                if duration_months < required_wait:
                    validation["violations"].append(
                        f"Maternity claims require {required_wait} months waiting period as per policy. Policy only {duration_months} months old."
                    )
                    validation["passed"] = False
                    validation["confidence_impact"] -= 0.4
        
        # Exclusion checking
        if parsed_query.get("condition"):
            condition = parsed_query["condition"].lower()
            exclusions = current_rules.get("exclusions", [])
            
            for exclusion in exclusions:
                if re.search(r'\b' + re.escape(exclusion.lower()) + r'\b', condition):
                    validation["violations"].append(f"Treatment '{condition}' contains excluded procedure: {exclusion}")
                    validation["passed"] = False
                    validation["confidence_impact"] -= 0.5
        
        return validation

# --- Multi-Factor Confidence Calculator ---
class ConfidenceCalculator:
    def __init__(self):
        pass
    
    def calculate_comprehensive_confidence(
        self, 
        query: str, 
        parsed_query: Dict,
        retrieved_docs: List,
        rule_validation: Dict,
        llm_response: Dict
    ) -> Dict[str, Any]:
        """Calculate confidence based on multiple factors"""
        
        factors = {}
        
        # 1. Query completeness (0-1)
        factors["query_completeness"] = parsed_query.get("completeness_score", 0.0)
        
        # 2. Document relevance (0-1)
        factors["doc_relevance"] = self.calculate_doc_relevance(query, retrieved_docs)
        
        # 3. Rule validation impact (-0.5 to +0.3)
        factors["rule_impact"] = max(-0.5, min(0.3, rule_validation.get("confidence_impact", 0.0)))
        
        # 4. LLM confidence (0-1)
        factors["llm_confidence"] = llm_response.get("confidence", 0.5)
        
        # 5. Response consistency (0-1)
        factors["response_consistency"] = self.calculate_response_consistency(llm_response)
        
        # Weighted calculation
        weights = {
            "query_completeness": 0.2,
            "doc_relevance": 0.25,
            "rule_impact": 0.2,
            "llm_confidence": 0.25,
            "response_consistency": 0.1
        }
        
        # Base confidence from weighted factors
        base_confidence = (
            factors["query_completeness"] * weights["query_completeness"] +
            factors["doc_relevance"] * weights["doc_relevance"] +
            factors["llm_confidence"] * weights["llm_confidence"] +
            factors["response_consistency"] * weights["response_consistency"]
        )
        
        # Apply rule impact
        final_confidence = base_confidence + (factors["rule_impact"] * weights["rule_impact"])
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return {
            "final_confidence": final_confidence,
            "factor_breakdown": factors,
            "weights_used": weights,
            "explanation": self.generate_confidence_explanation(factors, final_confidence)
        }
    
    def calculate_doc_relevance(self, query: str, docs: List) -> float:
        """Calculate document relevance score"""
        if not docs:
            return 0.0
        
        query_words = set(query.lower().split())
        total_relevance = 0.0
        
        for doc in docs:
            doc_text = doc.page_content.lower() if hasattr(doc, 'page_content') else str(doc).lower()
            doc_words = set(doc_text.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(doc_words))
            relevance = overlap / len(query_words) if query_words else 0.0
            total_relevance += relevance
        
        return min(1.0, total_relevance / len(docs))
    
    def calculate_response_consistency(self, response: Dict) -> float:
        """Check if LLM response is internally consistent"""
        decision = response.get("decision", "").upper()
        amount = response.get("amount", 0)
        justification = response.get("justification", "")
        
        consistency_score = 1.0
        
        # Check decision-amount consistency
        if decision == "APPROVED":
            if isinstance(amount, (int, float)) and amount == 0:
                consistency_score -= 0.3
        elif decision == "REJECTED":
            if isinstance(amount, (int, float)) and amount != 0:
                consistency_score -= 0.4
        
        # Check justification length and relevance
        if len(justification) < 20:
            consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def generate_confidence_explanation(self, factors: Dict, final_confidence: float) -> str:
        """Generate human-readable confidence explanation"""
        explanations = []
        
        if factors["query_completeness"] > 0.7:
            explanations.append("✓ Query contains most required information")
        elif factors["query_completeness"] > 0.4:
            explanations.append("⚠ Query missing some details")
        else:
            explanations.append("✗ Query lacks important information")
        
        if factors["doc_relevance"] > 0.6:
            explanations.append("✓ Found highly relevant policy documents")
        elif factors["doc_relevance"] > 0.3:
            explanations.append("⚠ Found moderately relevant documents")
        else:
            explanations.append("✗ Limited relevant documentation found")
        
        if factors["rule_impact"] > 0.1:
            explanations.append("✓ Passes insurance rule validations")
        elif factors["rule_impact"] > -0.1:
            explanations.append("⚠ Some rule concerns identified")
        else:
            explanations.append("✗ Violates key insurance rules")
        
        return " | ".join(explanations)

# --- Enhanced Document Processing with Source Citations ---
@st.cache_resource
def create_enhanced_vector_store(file_path: str) -> Tuple[FAISS, Dict, Dict]:
    """Create vector store with page number tracking AND extract policy rules"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading document...")
    progress_bar.progress(25)
    
    # Load with page tracking
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Extract full text for rule extraction
    full_text = "\n".join([doc.page_content for doc in docs])
    
    status_text.text("Processing pages and creating chunks...")
    progress_bar.progress(50)
    
    # Enhanced text splitter with metadata preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # Create page mapping
    page_mapping = {}
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get('page', 0) + 1  # Convert to 1-based indexing
        page_mapping[i] = page_num
        # Add page info to chunk metadata
        chunk.metadata['page_number'] = page_num
        chunk.metadata['chunk_id'] = i
    
    status_text.text("Creating embeddings...")
    progress_bar.progress(75)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    
    # Extract policy-specific rules
    status_text.text("Extracting policy rules...")
    progress_bar.progress(90)
    
    rule_engine = InsuranceRuleEngine()
    extracted_rules = rule_engine.extract_rules_from_policy(full_text)
    
    progress_bar.progress(100)
    status_text.text(f"✅ Processed {len(chunks)} chunks from {len(docs)} pages. Extracted {len(extracted_rules)} rule categories.")
    
    return vs, page_mapping, extracted_rules

# --- Enhanced Prompt Template with Source Citations ---
def create_enhanced_prompt_template() -> Tuple[PromptTemplate, StructuredOutputParser]:
    """Enhanced prompt with source citation requirements"""
    schemas = [
        ResponseSchema(name="decision", description="APPROVED, REJECTED, or REQUIRES_CLARIFICATION"),
        ResponseSchema(name="amount", description="Payout amount if approved; 'Not specified in policy' if coverage exists but amount not mentioned; 'As per network rates' for cashless; 0 if rejected"),
        ResponseSchema(name="justification", description="Exact clause or policy text supporting the decision with page reference"),
        ResponseSchema(name="confidence", description="AI confidence score as decimal between 0.0 and 1.0"),
        ResponseSchema(name="reasoning_steps", description="Step-by-step reasoning process as a list"),
        ResponseSchema(name="source_pages", description="List of page numbers where supporting information was found"),
        ResponseSchema(name="rule_violations", description="List of any insurance rule violations identified")
    ]
    
    parser = StructuredOutputParser.from_response_schemas(schemas)
    
    prompt_text = '''You are an expert insurance claims analyst AI with deep knowledge of insurance policies and regulations.

HALLUCINATION PREVENTION:
- You MUST NOT invent or infer policy clauses.
- If the policy does not explicitly cover a condition, you MUST state that coverage is not found.
- DO NOT associate a condition with a seemingly related but incorrect policy clause (e.g., do not associate 'knee surgery' with 'genitourinary surgery').
- If you are unsure, your decision MUST be REQUIRES_CLARIFICATION.

CRITICAL ANALYSIS FRAMEWORK:
1. Analyze the claim against EXPLICIT policy benefits only
2. Check for rule violations (waiting periods, exclusions, age limits)
3. Provide step-by-step reasoning for transparency
4. Include exact page references for all supporting evidence

CONTEXT FROM POLICY DOCUMENTS:
{context}

CLAIM QUERY:
{question}

DECISION CRITERIA:
- APPROVED: Specific treatment/service explicitly covered AND no rule violations. The covered service MUST be directly related to the patient's condition.
- REJECTED: Treatment not covered OR rule violations present  
- REQUIRES_CLARIFICATION: Insufficient information to make definitive decision, or ambiguity in the policy.

SOURCE CITATION REQUIREMENTS:
- Every justification MUST include page number reference
- Format: "As per policy clause X on page Y..."
- Multiple sources: "Based on pages X, Y, and Z..."
- No justification without page reference

REASONING PROCESS:
1. Parse claim details (age, condition, policy duration, etc.)
2. Check rule violations (waiting periods, exclusions, age limits)
3. Search for relevant policy coverage for the EXACT condition mentioned in the query.
4. Match claim to specific policy benefits.
5. Determine coverage amount based on policy terms.
6. Provide final decision with confidence assessment.

RESPONSE REQUIREMENTS:
- Use ONLY information from provided context.
- Include page numbers for ALL supporting evidence.
- Provide detailed step-by-step reasoning
- List any rule violations found
- Return valid JSON without comments

{format_instructions}

CRITICAL: Every justification must cite specific page numbers. Never approve without explicit policy coverage evidence.'''

    return PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    ), parser

# --- Enhanced QA Chain with Source Tracking ---
def create_enhanced_qa_chain(vs: FAISS, page_mapping: Dict):
    """Create QA chain with enhanced source tracking"""
    llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)
    prompt, parser = create_enhanced_prompt_template()
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vs.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain, parser

# --- Response Processing with Enhanced Error Handling ---
def process_enhanced_response(response: str, parser, rule_validation: Dict) -> Tuple[Dict[str, Any], str]:
    """Enhanced response processing with rule integration"""
    try:
        # Clean response
        cleaned_response = clean_json_response(response)
        
        # Try structured parsing first
        try:
            parsed = parser.parse(cleaned_response)
            parse_method = "structured"
        except Exception as e:
            # Enhanced fallback to JSON extraction
            try:
                # Attempt direct JSON parsing first
                parsed = json.loads(cleaned_response)
                parse_method = "direct_json"
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        parse_method = "json_extraction"
                    except json.JSONDecodeError:
                        # Final fallback with detailed error
                        error_msg = f"JSON parsing failed: {str(e)}. Raw response: {cleaned_response[:500]}..."
                        raise ValueError(error_msg)
                else:
                    raise ValueError("No JSON structure found in response")
        
        # Integrate rule validation results
        if rule_validation.get("violations"):
            if parsed.get("decision") == "APPROVED":
                parsed["decision"] = "REJECTED"
                parsed["rule_violations"] = rule_validation["violations"]
                parsed["justification"] = f"RULE VIOLATION: {'; '.join(rule_validation['violations'])}. " + parsed.get("justification", "")
        
        # Ensure required fields
        required_fields = {
            "decision": "REQUIRES_CLARIFICATION",
            "amount": 0,
            "justification": "No justification provided",
            "confidence": 0.5,
            "reasoning_steps": [],
            "source_pages": [],
            "rule_violations": rule_validation.get("violations", [])
        }
        
        for field, default in required_fields.items():
            if field not in parsed:
                parsed[field] = default
        
        # Normalize data types
        if isinstance(parsed.get("amount"), str):
            amt_str = parsed["amount"].replace(",", "").strip()
            if amt_str.isdigit():
                parsed["amount"] = int(amt_str)
            else:
                # Keep as string for non-numeric amounts
                parsed["amount"] = amt_str
        elif not isinstance(parsed.get("amount"), (int, float)):
            parsed["amount"] = 0
        
        try:
            if isinstance(parsed.get("confidence"), str):
                parsed["confidence"] = float(parsed["confidence"])
        except (ValueError, TypeError):
            parsed["confidence"] = 0.5
        
        # Ensure source_pages is always a list of ints
        if isinstance(parsed.get("source_pages"), str):
            page_nums = re.findall(r'page\s*(\d+)', parsed["source_pages"], re.IGNORECASE)
            try:
                parsed["source_pages"] = [int(p) for p in page_nums]
            except (ValueError, TypeError):
                parsed["source_pages"] = []
        elif not isinstance(parsed.get("source_pages"), list):
            parsed["source_pages"] = []
        
        # Convert reasoning_steps to list if string
        if isinstance(parsed.get("reasoning_steps"), str):
            # Split by numbered steps
            steps = re.split(r'\n\d+\.\s*', parsed["reasoning_steps"])
            parsed["reasoning_steps"] = [step.strip() for step in steps if step.strip()]
        elif not isinstance(parsed.get("reasoning_steps"), list):
            parsed["reasoning_steps"] = []
        
        # Ensure for REJECTED, amount is 0
        if parsed.get("decision") == "REJECTED":
            parsed["amount"] = 0
        
        return parsed, parse_method
        
    except Exception as e:
        return {
            "error": f"Parsing failed: {str(e)}",
            "raw_response": response,
            "decision": "ERROR",
            "amount": 0,
            "justification": "System error occurred",
            "confidence": 0.0,
            "reasoning_steps": ["System error during processing"],
            "source_pages": [],
            "rule_violations": []
        }, "error"

def clean_json_response(response: str) -> str:
    """Clean JSON response"""
    response = re.sub(r'//.*?(?=\n|$)', '', response)
    response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
    response = re.sub(r',(\s*[}\]])', r'\1', response)
    return response.strip()

# --- Batch Processing System ---
class BatchProcessor:
    def __init__(self, qa_chain, parser, query_parser, rule_engine, confidence_calc, page_mapping):
        self.qa_chain = qa_chain
        self.parser = parser
        self.query_parser = query_parser
        self.rule_engine = rule_engine
        self.confidence_calc = confidence_calc
        self.page_mapping = page_mapping
    
    def process_batch(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries efficiently"""
        results = []
        
        for i, query in enumerate(queries):
            try:
                # Process single query
                result = self.process_single_query(query)
                result["batch_index"] = i
                result["query"] = query
                results.append(result)
                
            except Exception as e:
                results.append({
                    "batch_index": i,
                    "query": query,
                    "error": str(e),
                    "decision": "ERROR"
                })
        
        return results
    
    def process_single_query(self, query: str) -> Dict:
        """Process a single query with full pipeline"""
        # Parse query
        parsed_query = self.query_parser.parse_query(query)
        
        # Rule validation
        rule_validation = self.rule_engine.validate_claim(parsed_query)
        
        # Get LLM response
        llm_result = self.qa_chain.invoke({"query": query})
        raw_response = llm_result["result"]
        retrieved_docs = llm_result.get("source_documents", [])
        
        # Process response
        processed_response, parse_method = process_enhanced_response(raw_response, self.parser, rule_validation)
        
        # Calculate comprehensive confidence
        if "error" not in processed_response:
            confidence_analysis = self.confidence_calc.calculate_comprehensive_confidence(
                query, parsed_query, retrieved_docs, rule_validation, processed_response
            )
            processed_response["final_confidence"] = confidence_analysis["final_confidence"]
            processed_response["confidence_breakdown"] = confidence_analysis["factor_breakdown"]
            processed_response["confidence_explanation"] = confidence_analysis["explanation"]
        
        # Add metadata
        processed_response["parsed_query"] = parsed_query
        processed_response["rule_validation"] = rule_validation
        processed_response["parse_method"] = parse_method
        processed_response["timestamp"] = datetime.now().isoformat()
        
        return processed_response

# --- Analytics Dashboard ---
def create_analytics_dashboard():
    """Comprehensive analytics dashboard"""
    if "interaction_log" not in st.session_state or not st.session_state.interaction_log:
        st.info("📊 Process some queries to see analytics")
        return
    
    st.header("📊 Decision Analytics Dashboard")
    
    # Prepare data
    logs = st.session_state.interaction_log
    df = pd.DataFrame([
        {
            "timestamp": pd.to_datetime(log.get("timestamp", datetime.now())),
            "decision": log.get("decision", "Unknown"),
            "confidence": log.get("final_confidence", log.get("confidence", 0)),
            "amount": log.get("amount", 0),
            "has_violations": len(log.get("rule_violations", [])) > 0,
            "completeness": log.get("parsed_query", {}).get("completeness_score", 0),
            "parse_method": log.get("parse_method", "unknown")
        }
        for log in logs
    ])
    
    # Add numeric amount column for calculations
    df['amount_numeric'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_queries = len(df)
        st.metric("Total Queries", total_queries)
    
    with col2:
        approval_rate = (df["decision"] == "APPROVED").mean() * 100 if total_queries > 0 else 0
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        avg_confidence = df["confidence"].mean() if total_queries > 0 else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        # Use numeric amount for approved claims
        approved_numeric = df[(df["decision"] == "APPROVED") & (df['amount_numeric'].notna())]['amount_numeric']
        if not approved_numeric.empty:
            avg_amount = approved_numeric.mean()
            st.metric("Avg Claim Amount", f"₹{avg_amount:,.0f}")
        else:
            st.metric("Avg Claim Amount", "N/A")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Decision Distribution
        decision_counts = df["decision"].value_counts()
        fig_decisions = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            title="Decision Distribution"
        )
        st.plotly_chart(fig_decisions, use_container_width=True)
    
    with col2:
        # Confidence Distribution
        fig_confidence = px.histogram(
            df, x="confidence", nbins=10,
            title="Confidence Score Distribution",
            labels={"confidence": "Confidence Score", "count": "Number of Queries"}
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Time Series Analysis
    if total_queries > 1:
        st.subheader("📈 Trends Over Time")
        
        # Resample by hour if we have enough data points
        df_time = df.set_index('timestamp').resample('H').agg({
            'decision': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=df_time['timestamp'], 
            y=df_time['decision'],
            mode='lines+markers',
            name='Queries per Hour',
            yaxis='y'
        ))
        fig_timeline.add_trace(go.Scatter(
            x=df_time['timestamp'], 
            y=df_time['confidence'],
            mode='lines+markers',
            name='Avg Confidence',
            yaxis='y2'
        ))
        
        fig_timeline.update_layout(
            title='Query Volume and Confidence Trends',
            xaxis_title='Time',
            yaxis=dict(title='Query Count', side='left'),
            yaxis2=dict(title='Confidence Score', side='right', overlaying='y'),
            hovermode='x unified'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Detailed Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Query Completeness Analysis")
        completeness_bins = pd.cut(df["completeness"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
        completeness_counts = completeness_bins.value_counts()
        
        fig_completeness = px.bar(
            x=completeness_counts.index,
            y=completeness_counts.values,
            title="Query Completeness Distribution",
            labels={"x": "Completeness Level", "y": "Count"}
        )
        st.plotly_chart(fig_completeness, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Rule Violations Analysis")
        violation_rate = df["has_violations"].mean() * 100
        st.metric("Queries with Rule Violations", f"{violation_rate:.1f}%")
        
        if violation_rate > 0:
            violation_impact = df[df["has_violations"]]["decision"].value_counts()
            fig_violations = px.bar(
                x=violation_impact.index,
                y=violation_impact.values,
                title="Impact of Rule Violations on Decisions"
            )
            st.plotly_chart(fig_violations, use_container_width=True)
    
    # Recent Queries Table
    st.subheader("📋 Recent Query Analysis")
    recent_df = df.tail(10)[["timestamp", "decision", "confidence", "amount", "has_violations"]].copy()
    recent_df["timestamp"] = recent_df["timestamp"].dt.strftime("%H:%M:%S")
    st.dataframe(recent_df, use_container_width=True)

# --- Reasoning Path Display ---
def display_reasoning_path(response: Dict):
    """Display detailed reasoning path"""
    st.subheader("🧠 AI Reasoning Path")
    
    # Query Analysis
    with st.expander("1️⃣ Query Analysis", expanded=False):
        parsed_query = response.get("parsed_query", {})
        if parsed_query:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Extracted Information:**")
                for key, value in parsed_query.items():
                    if value and key != "completeness_score":
                        st.write(f"• {key.replace('_', ' ').title()}: {value}")
            
            with col2:
                completeness = parsed_query.get("completeness_score", 0)
                st.metric("Query Completeness", f"{completeness:.1%}")
                
                missing_fields = st.session_state.query_parser.get_missing_fields(parsed_query)
                
                if missing_fields:
                    st.warning(f"Missing: {', '.join(missing_fields)}")
    
    # Rule Validation
    with st.expander("2️⃣ Rule Validation", expanded=False):
        rule_validation = response.get("rule_validation", {})
        if rule_validation:
            if rule_validation.get("violations"):
                st.error("**Rule Violations Found:**")
                for violation in rule_validation["violations"]:
                    st.write(f"❌ {violation}")
            
            if rule_validation.get("warnings"):
                st.warning("**Warnings:**")
                for warning in rule_validation["warnings"]:
                    st.write(f"⚠️ {warning}")
            
            if rule_validation.get("applicable_rules"):
                st.success("**Applicable Rules:**")
                for rule in rule_validation["applicable_rules"]:
                    st.write(f"✅ {rule}")
    
    # AI Reasoning Steps
    with st.expander("3️⃣ AI Analysis Steps", expanded=True):
        reasoning_steps = response.get("reasoning_steps", [])
        if reasoning_steps:
            for i, step in enumerate(reasoning_steps, 1):
                st.write(f"**Step {i}:** {step}")
        else:
            st.write("No detailed reasoning steps available")
    
    # Source Citations
    with st.expander("4️⃣ Source Citations", expanded=False):
        source_pages = response.get("source_pages", [])
        justification = response.get("justification", "")
        
        col1, col2 = st.columns(2)
        with col1:
            if source_pages:
                st.write("**Referenced Pages:**")
                for page in source_pages:
                    st.write(f"📄 Page {page}")
            else:
                st.write("No specific page references found")
        
        with col2:
            st.write("**Supporting Evidence:**")
            st.write(justification)
    
    # Confidence Breakdown
    with st.expander("5️⃣ Confidence Analysis", expanded=False):
        confidence_breakdown = response.get("confidence_breakdown", {})
        confidence_explanation = response.get("confidence_explanation", "")
        
        if confidence_breakdown:
            st.write("**Confidence Factors:**")
            for factor, score in confidence_breakdown.items():
                if isinstance(score, (int, float)):
                    st.write(f"• {factor.replace('_', ' ').title()}: {score:.2f}")
        
        if confidence_explanation:
            st.info(confidence_explanation)

# --- JSON Output Display ---
def display_json_output(response: Dict):
    """Display JSON output in a formatted box"""
    st.subheader("📄 JSON Output")
    
    # Create clean JSON structure
    json_output = {
        "Decision": response.get("decision", "Unknown"),
        "Amount": response.get("amount", 0),
        "Justification": response.get("justification", ""),
        "Confidence": response.get("final_confidence", response.get("confidence", 0.0)),
        "RuleViolations": response.get("rule_violations", []),
        "SourcePages": response.get("source_pages", [])
    }
    
    # Display in a formatted box
    st.markdown(f"```json\n{json.dumps(json_output, indent=2, ensure_ascii=False)}\n```")

def render_chat_message(message: Dict):
    """Render a single chat message."""
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            response = message["content"]
            if isinstance(response, dict) and "error" not in response:
                # Display main decision
                decision = response.get("decision", "Unknown")
                confidence = response.get("final_confidence", response.get("confidence", 0))

                # Decision display with color coding
                if decision == "APPROVED":
                    st.success(f"✅ **APPROVED** (Confidence: {confidence:.1%})")
                elif decision == "REJECTED":
                    st.error(f"❌ **REJECTED** (Confidence: {confidence:.1%})")
                else:
                    st.warning(f"⚠️ **{decision}** (Confidence: {confidence:.1%})")

                # Amount and justification
                amount = response.get("amount", 0)
                if amount and amount != 0:
                    if isinstance(amount, (int, float)) and amount > 0:
                        st.info(f"💰 **Amount:** ₹{amount:,}")
                    else:
                        st.info(f"💰 **Amount:** {amount}")

                justification = response.get("justification", "")
                if justification:
                    st.markdown(f"**Justification:** {justification}")

                # Show rule violations if any
                violations = response.get("rule_violations", [])
                if violations:
                    st.error("**Rule Violations:**")
                    for violation in violations:
                        st.write(f"• {violation}")

                # Display reasoning path
                display_reasoning_path(response)

                # Display JSON output
                display_json_output(response)

            else:
                st.error("Error processing query")
                if "raw_response" in response:
                    with st.expander("Raw Response"):
                        st.code(response["raw_response"])
        else:
            st.markdown(message["content"])

# --- Main Streamlit Application ---
def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📄 Decision Co-Pilot: AI-Powered Insurance Claims Assistant")
    st.markdown("*Enhanced with Smart Query Processing, Rule Validation & Source Citations*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize components
    if "query_parser" not in st.session_state:
        st.session_state.query_parser = QueryParser()
    if "rule_engine" not in st.session_state:
        st.session_state.rule_engine = InsuranceRuleEngine()
    if "confidence_calc" not in st.session_state:
        st.session_state.confidence_calc = ConfidenceCalculator()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_file = st.file_uploader(
            "Upload Insurance Policy PDF", 
            type="pdf",
            help="Upload your insurance policy document for analysis"
        )
        
        if uploaded_file:
            # Process uploaded file
            os.makedirs(TEMP_DIR, exist_ok=True)
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Create enhanced vector store
            vector_store, page_mapping, extracted_rules = create_enhanced_vector_store(file_path)
            qa_chain, output_parser = create_enhanced_qa_chain(vector_store, page_mapping)
            
            # Store in session state
            st.session_state.qa_chain = qa_chain
            st.session_state.parser = output_parser
            st.session_state.page_mapping = page_mapping
            st.session_state.document_name = uploaded_file.name
            
            # Initialize batch processor
            st.session_state.batch_processor = BatchProcessor(
                qa_chain, output_parser, st.session_state.query_parser,
                st.session_state.rule_engine, st.session_state.confidence_calc, page_mapping
            )
            # Update rule engine with extracted rules
            st.session_state.rule_engine.extracted_rules = extracted_rules
    
            # Show extracted rules in sidebar
            if extracted_rules:
                st.sidebar.success(f"✅ Extracted {len(extracted_rules)} policy-specific rules")
                with st.sidebar.expander("📋 Policy Rules Detected"):
                    for category, rules in extracted_rules.items():
                        st.write(f"**{category.replace('_', ' ').title()}:**")
                        if isinstance(rules, dict):
                            for key, value in rules.items():
                                st.write(f"  • {key}: {value}")
                        elif isinstance(rules, list):
                            for rule in rules:
                                st.write(f"  • {rule}")
                        else:
                            st.write(f"  • {rules}")
            else:
                st.sidebar.warning("⚠️ Using default rules - no policy-specific rules found")
        
        # System Status
        st.header("⚡ System Status")
        if "qa_chain" in st.session_state:
            st.success(f"✅ Document: {st.session_state.get('document_name', 'Loaded')}")
            st.success("✅ AI Model: Ready")
            st.success("✅ Rule Engine: Active")
        else:
            st.warning("⏳ Upload document to activate")
    
    # Main content area
    if "qa_chain" in st.session_state:
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["💬 Single Query", "📦 Batch Processing", "📊 Analytics"])
        
        with tab1:
            st.subheader("Single Query Analysis")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                render_chat_message(message)
            
            # Query input
            query = st.chat_input("Enter your insurance claim query (e.g., '46-year-old male, knee surgery in Pune, 3-month-old policy')...")
            
            if query:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": query})
                
                with st.chat_message("user"):
                    st.markdown(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Analyzing query and processing claim..."):
                        # Process query with full pipeline
                        try:
                            result = st.session_state.batch_processor.process_single_query(query)
                            
                            # Log interaction
                            if "interaction_log" not in st.session_state:
                                st.session_state.interaction_log = []
                            st.session_state.interaction_log.append(result)
                            
                            # Display result
                            decision = result.get("decision", "Unknown")
                            confidence = result.get("final_confidence", result.get("confidence", 0))
                            
                            # Decision display
                            if decision == "APPROVED":
                                st.success(f"✅ **APPROVED** (Confidence: {confidence:.1%})")
                            elif decision == "REJECTED":
                                st.error(f"❌ **REJECTED** (Confidence: {confidence:.1%})")
                            else:
                                st.warning(f"⚠️ **{decision}** (Confidence: {confidence:.1%})")
                            
                            # Amount and justification
                            amount = result.get("amount", 0)
                            if amount and amount != 0:
                                if isinstance(amount, (int, float)) and amount > 0:
                                    st.info(f"💰 **Amount:** ₹{amount:,}")
                                else:
                                    st.info(f"💰 **Amount:** {amount}")
                            
                            justification = result.get("justification", "")
                            if justification:
                                st.markdown(f"**Justification:** {justification}")
                            
                            # Show rule violations if any
                            violations = result.get("rule_violations", [])
                            if violations:
                                st.error("**Rule Violations:**")
                                for violation in violations:
                                    st.write(f"• {violation}")
                            
                            # Display reasoning path
                            display_reasoning_path(result)
                            
                            # Display JSON output
                            display_json_output(result)
                            
                            # Add to message history
                            st.session_state.messages.append({"role": "assistant", "content": result})
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
        
        with tab2:
            st.subheader("Batch Query Processing")
            st.markdown("Process multiple queries simultaneously for efficiency")
            
            # Batch input
            batch_input = st.text_area(
                "Enter multiple queries (one per line):",
                height=200,
                placeholder="""46-year-old male, knee surgery in Pune, 3-month-old policy
30-year-old female, diabetes treatment, 2-year-old policy
Accidental injury claim, broken arm, 6-month policy
Maternity delivery claim, policy purchased 10 months ago""",
                help="Enter each query on a separate line. The system will process all queries and provide structured results."
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                process_batch = st.button("🚀 Process Batch", type="primary")
            with col2:
                if batch_input:
                    query_count = len([q.strip() for q in batch_input.split('\n') if q.strip()])
                    st.info(f"Ready to process {query_count} queries")
            
            if process_batch and batch_input:
                queries = [q.strip() for q in batch_input.split('\n') if q.strip()]
                
                if queries:
                    with st.spinner(f"Processing {len(queries)} queries..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        batch_results = []
                        for i, query in enumerate(queries):
                            status_text.text(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
                            
                            try:
                                result = st.session_state.batch_processor.process_single_query(query)
                                batch_results.append(result)
                                
                                # Log to interaction history
                                if "interaction_log" not in st.session_state:
                                    st.session_state.interaction_log = []
                                st.session_state.interaction_log.append(result)
                                
                            except Exception as e:
                                batch_results.append({
                                    "query": query,
                                    "error": str(e),
                                    "decision": "ERROR",
                                    "confidence": 0.0
                                })
                            
                            progress_bar.progress((i + 1) / len(queries))
                        
                        status_text.text("✅ Batch processing complete!")
                        
                        # Display results
                        st.subheader("📋 Batch Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        approved_count = sum(1 for r in batch_results if r.get("decision") == "APPROVED")
                        rejected_count = sum(1 for r in batch_results if r.get("decision") == "REJECTED")
                        error_count = sum(1 for r in batch_results if r.get("decision") == "ERROR")
                        avg_confidence = np.mean([r.get("final_confidence", r.get("confidence", 0)) for r in batch_results])
                        
                        with col1:
                            st.metric("Approved", approved_count)
                        with col2:
                            st.metric("Rejected", rejected_count)
                        with col3:
                            st.metric("Errors", error_count)
                        with col4:
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        # Detailed results table
                        results_df = pd.DataFrame([
                            {
                                "Query": r.get("query", "")[:100] + "..." if len(r.get("query", "")) > 100 else r.get("query", ""),
                                "Decision": r.get("decision", "Unknown"),
                                "Amount": str(r.get("amount", 0)),
                                "Confidence": f"{r.get('final_confidence', r.get('confidence', 0)):.2%}",
                                "Rule Violations": len(r.get("rule_violations", [])),
                                "Source Pages": ", ".join(map(str, r.get("source_pages", []))) if r.get("source_pages") else "None"
                            }
                            for r in batch_results
                        ])
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Export option
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Display JSON outputs for each query
                        st.subheader("🧾 JSON Outputs")
                        for i, result in enumerate(batch_results):
                            with st.expander(f"Query {i+1}: {result.get('query', 'Unknown Query')[:50]}...", expanded=False):
                                if "error" in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    # Create clean JSON structure
                                    json_output = {
                                        "Decision": result.get("decision", "Unknown"),
                                        "Amount": result.get("amount", 0),
                                        "Justification": result.get("justification", ""),
                                        "Confidence": result.get("final_confidence", result.get("confidence", 0.0)),
                                        "RuleViolations": result.get("rule_violations", []),
                                        "SourcePages": result.get("source_pages", [])
                                    }
                                    st.markdown(f"```json\n{json.dumps(json_output, indent=2, ensure_ascii=False)}\n```")
        
        with tab3:
            create_analytics_dashboard()
    
    else:
        # Welcome screen
        st.info("👆 Please upload an insurance policy PDF document to start analyzing claims")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 **Enhanced Features:**
            - **Smart Query Processing** - Extracts structured information
            - **Rule-Based Validation** - Insurance domain expertise
            - **Source Citations** - Page-level references  
            - **Multi-Factor Confidence** - Comprehensive scoring
            - **Batch Processing** - Handle multiple queries
            - **Analytics Dashboard** - Decision insights
            - **Reasoning Transparency** - Step-by-step analysis
            """)
        
        with col2:
            st.markdown("""
            ### 📋 **Example Queries:**
            - "46-year-old male, knee surgery in Pune, 3-month-old policy"
            - "Pre-existing diabetes, 30-year-old female, policy 2 years old"
            - "Accidental injury claim, broken arm, 6-month policy"
            - "Maternity delivery, policy purchased 10 months ago"
            - "Cardiac surgery claim, 55-year-old male, comprehensive policy"
            """)
        
        st.markdown("---")
        st.markdown("*Built with Advanced RAG Pipeline + Ollama Llama 3 + LangChain + FAISS + Streamlit*")

if __name__ == "__main__":
    main()