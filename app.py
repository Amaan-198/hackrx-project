# Enhanced Decision Co-Pilot with Complete Feature Set
import streamlit as st
import os
import json
import re
from typing import Dict, Any, List, Tuple

# Fix HuggingFace environment issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import copy
import hashlib

# --- Configuration ---
TEMP_DIR = "temp"
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Enhanced UI Configuration ---
st.set_page_config(
    page_title="Decision Co-Pilot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    .stSpinner > div { text-align: center; }
    .stDownloadButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- Smart Query Preprocessing ---
class QueryParser:
    def __init__(self):
        self.nlp = None  # spaCy model not used; all parsing is regex-based
    
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
            r'policy\s*(?:is\s*)?(\d+)[-\s]?(month|year)s?\s*old',
            r'(\d+)[-\s]?(month|year)s?\s*policy',
            r'policy\s*for\s*(\d+)[-\s]?(month|year)s?',
            r'policy\s+(?:purchased|bought|taken|started|active)\s+(\d+)\s*(month|year)s?\s*ago',
        ]


        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Extract and REMOVE this match from query
                parsed["policy_duration"] = int(match.group(1))
                parsed["policy_duration_unit"] = match.group(2).lower() if match.group(2) else "month"
                parsed["completeness_score"] += 0.15
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
                    parsed["completeness_score"] += 0.15
                except (ValueError, TypeError):
                    pass
                break
        # Extract gender — check female-specific words first to avoid
        # substring false-positives (e.g. "female" contains "male",
        # "claim" contains the letter "m").
        # Use regex word boundaries so we only match whole tokens.
        if re.search(r'\bfemale\b|\bwoman\b|\blady\b|\bgirl\b', query_lower):
            parsed["gender"] = "female"
            parsed["completeness_score"] += 0.15
        elif re.search(r'\bmale\b|\bman\b|\bgentleman\b', query_lower):
            parsed["gender"] = "male"
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
            r"exclusions?:?\s*([^.]+)",
            r"not[-\s]?covered:?\s*([^.]+)",
            r"excluded[-\s]?treatments?:?\s*([^.]+)",
            r"conditions?\s*not\s*covered:?\s*([^.]+)"
        ]
        
        exclusions = []
        for pattern in exclusion_patterns:
            matches = re.findall(pattern, policy_lower, re.MULTILINE | re.DOTALL)
            for match in matches:
                # Clean and split exclusions
                items = [item.strip() for item in re.split(r'[,\n•]', match) if item.strip()]
                exclusions.extend(items)
        
        if exclusions:
            rules["exclusions"] = exclusions
        
        self.extracted_rules = rules
        return rules
    
    def get_applicable_rules(self) -> Dict:
        """Get rules with policy-specific overriding defaults"""
        final_rules = copy.deepcopy(self.default_rules)
        
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
                "warnings": [],
                "applicable_rules": [],
                "confidence_impact": 0.0,
                "rules_source": "skipped_invalid_age"
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

# --- Model Initialization Helper ---
@st.cache_resource
def initialize_embedding_model():
    """Initialize and cache the embedding model"""
    try:
        import os
        # Force online mode
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )
        return embeddings, None
    except Exception as e:
        error_msg = str(e)
        
        troubleshooting = """
**🔧 Troubleshooting Steps:**

1. **Test Internet Connection:**
   ```
   curl -I https://huggingface.co
   ```

2. **Check Proxy Settings (if behind corporate firewall):**
   ```
   set HTTP_PROXY=http://your-proxy:port
   set HTTPS_PROXY=http://your-proxy:port
   ```

3. **Manual Model Download:**
   Open a command prompt and run:
   ```
   cd d:\\projects\\hackrx-project
   .\\env\\Scripts\\activate
   python -c "import os; os.environ['HF_HUB_OFFLINE']='0'; from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
   ```

4. **Offline Mode (if model already downloaded):**
   Set environment variable before running:
   ```
   set HF_HUB_OFFLINE=1
   ```

5. **Use Alternative Model Path:**
   If you have the model files locally, update EMBEDDING_MODEL in app.py to point to local path
"""
        
        if "couldn't connect" in error_msg.lower() or "connection" in error_msg.lower():
            return None, f"❌ **Cannot connect to HuggingFace servers**\n\n{error_msg}\n\n{troubleshooting}"
        else:
            return None, f"❌ **Error loading embedding model**\n\n{error_msg}\n\n{troubleshooting}"

# --- Enhanced Document Processing with Source Citations ---
@st.cache_resource
def create_enhanced_vector_store(file_path: str, file_hash: str = "") -> Tuple[FAISS, Dict, Dict]:
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
    
    # Use pre-initialized embedding model
    embeddings, error = initialize_embedding_model()
    if error:
        st.error(error)
        st.info("💡 **Troubleshooting Steps:**\n1. Ensure you have stable internet connection\n2. Check firewall/proxy settings\n3. Try running: `pip install --upgrade sentence-transformers`\n4. The model downloads to: `~/.cache/huggingface/`")
        raise Exception(error)
    
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
- APPROVED: Specific treatment/service explicitly covered AND no rule violations
- REJECTED: Treatment not covered OR rule violations present  
- REQUIRES_CLARIFICATION: Insufficient information to make definitive decision

SOURCE CITATION REQUIREMENTS:
- Every justification MUST include page number reference
- Format: "As per policy clause X on page Y..."
- Multiple sources: "Based on pages X, Y, and Z..."
- No justification without page reference

REASONING PROCESS:
1. Parse claim details (age, condition, policy duration, etc.)
2. Check rule violations (waiting periods, exclusions, age limits)
3. Search for relevant policy coverage
4. Match claim to specific policy benefits
5. Determine coverage amount based on policy terms
6. Provide final decision with confidence assessment

RESPONSE REQUIREMENTS:
- Use ONLY information from provided context
- Include page numbers for ALL supporting evidence
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
                    raise ValueError("No JSON structure found in response")#
        
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
            if isinstance(parsed.get("confidence"), (int, float)):
                parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
        except (ValueError, TypeError):
            parsed["confidence"] = 0.5
        
        # Ensure source_pages is always a list of ints
        if isinstance(parsed.get("source_pages"), str):
            page_nums = re.findall(r'page\s*(\d+)', parsed["source_pages"], re.IGNORECASE)
            parsed["source_pages"] = [int(p) for p in page_nums if p.isdigit()]
        elif isinstance(parsed.get("source_pages"), list):
            normalized = []
            for item in parsed["source_pages"]:
                if isinstance(item, int):
                    normalized.append(item)
                elif isinstance(item, float):
                    normalized.append(int(item))
                elif isinstance(item, str):
                    # Try "page N" pattern first (e.g. "page 7", "Page 12", "Section 2 page 7")
                    page_match = re.search(r'page\s*(\d+)', item, re.IGNORECASE)
                    if page_match:
                        normalized.append(int(page_match.group(1)))
                    else:
                        # Fall back to first digit sequence (e.g. "7", "p.7", "7.")
                        digit_match = re.search(r'\d+', item)
                        if digit_match:
                            normalized.append(int(digit_match.group()))
            parsed["source_pages"] = normalized
        else:
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

        # Normalize rule_violations to a list of strings
        rv = parsed.get("rule_violations", [])
        if isinstance(rv, str):
            rv = [rv] if rv.strip() else []
        elif not isinstance(rv, list):
            rv = []
        parsed["rule_violations"] = [str(v) for v in rv if v]

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
    """Clean JSON response by removing JS-style comments and trailing commas.

    Line comments (//) are only stripped when they appear outside of
    quoted strings so that URLs (e.g. https://…) inside values are preserved.
    """
    # --- Remove line comments (//) only when outside string literals ---
    lines = response.split('\n')
    cleaned_lines = []
    for line in lines:
        in_string = False
        escape_next = False
        cut_index = len(line)
        for i, c in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string and c == '/' and i + 1 < len(line) and line[i + 1] == '/':
                cut_index = i
                break
        cleaned_lines.append(line[:cut_index])
    response = '\n'.join(cleaned_lines)
    # --- Remove block comments (/* ... */) ---
    response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
    # --- Remove trailing commas before } or ] ---
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
            "amount": str(log.get("amount", 0)),
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
        completeness_bins = pd.cut(df["completeness"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
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


# --- Compact Response Renderer ---
def render_response(response: Dict, compact: bool = False):
    """Render a claim analysis response. Compact mode omits the details expander."""
    if not isinstance(response, dict):
        st.error("Invalid response data")
        return

    decision = response.get("decision", "Unknown")
    confidence = response.get("final_confidence", response.get("confidence", 0))
    try:
        confidence = float(confidence) if confidence is not None else 0.0
    except (ValueError, TypeError):
        confidence = 0.0
    amount = response.get("amount", 0)
    justification = response.get("justification", "")
    violations = response.get("rule_violations", [])
    source_pages = response.get("source_pages", [])

    # Error state
    if decision == "ERROR":
        st.error(f"⚠️ {response.get('error', 'Processing error')}")
        return

    # Decision badge
    if decision == "APPROVED":
        st.success(f"✅ **APPROVED** — Confidence: {confidence:.0%}")
    elif decision == "REJECTED":
        st.error(f"❌ **REJECTED** — Confidence: {confidence:.0%}")
    else:
        st.warning(f"ℹ️ **{decision}** — Confidence: {confidence:.0%}")

    # Amount (only if meaningful)
    if isinstance(amount, (int, float)) and amount > 0:
        st.markdown(f"💰 **Amount:** ₹{amount:,}")
    elif isinstance(amount, str) and amount not in ("0", ""):
        st.markdown(f"💰 **Amount:** {amount}")

    # Justification
    if justification:
        st.markdown(f"**Justification:** {justification}")

    # Violations (inline) — guard against string values
    if violations:
        if isinstance(violations, str):
            violations = [violations]
        if isinstance(violations, list) and violations:
            for v in violations:
                st.markdown(f"⚠️ _{v}_")

    # Source pages
    if source_pages:
        st.caption(f"📄 Pages: {', '.join(str(p) for p in source_pages)}")

    # Single details expander (hidden in compact mode to keep history clean)
    if not compact:
        with st.expander("View Details", expanded=False):
            # Reasoning steps
            steps = response.get("reasoning_steps", [])
            if steps:
                st.markdown("**Reasoning Steps:**")
                for i, step in enumerate(steps, 1):
                    clean_step = re.sub(
                        r'^\s*(?:step\s*)?\d+[.):]\s+', '', str(step), flags=re.IGNORECASE
                    )
                    st.markdown(f"{i}. {clean_step}")
                st.markdown("---")

            # JSON output
            st.json({
                "decision": decision,
                "amount": amount,
                "justification": justification,
                "confidence": round(confidence, 4) if isinstance(confidence, float) else confidence,
                "rule_violations": violations,
                "source_pages": source_pages,
                "reasoning_steps": steps,
            })


# --- Main Application ---
def main():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("🔑 **GROQ_API_KEY is not set.** Add it to your `.env` file and restart.")
        st.stop()

    st.title("📄 Decision Co-Pilot")
    st.caption("AI-Powered Insurance Claims Analysis · Groq LLaMA 3.3 70B")

    # --- Embedding model init ---
    if "embedding_check_done" not in st.session_state:
        with st.spinner("Initializing embedding model…"):
            embeddings, error = initialize_embedding_model()
            if error:
                st.error(error)
                st.stop()
            st.session_state.embedding_check_done = True

    # --- Components ---
    if "query_parser" not in st.session_state:
        st.session_state.query_parser = QueryParser()
    if "rule_engine" not in st.session_state:
        st.session_state.rule_engine = InsuranceRuleEngine()
    if "confidence_calc" not in st.session_state:
        st.session_state.confidence_calc = ConfidenceCalculator()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interaction_log" not in st.session_state:
        st.session_state.interaction_log = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("📁 Upload Policy")
        uploaded_file = st.file_uploader("Insurance Policy PDF", type="pdf")

        if uploaded_file:
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            if st.session_state.get("_file_hash") != file_hash:
                os.makedirs(TEMP_DIR, exist_ok=True)
                file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                vector_store, page_mapping, extracted_rules = create_enhanced_vector_store(file_path, file_hash)
                qa_chain, output_parser = create_enhanced_qa_chain(vector_store, page_mapping)

                st.session_state.qa_chain = qa_chain
                st.session_state.parser = output_parser
                st.session_state.page_mapping = page_mapping
                st.session_state.document_name = uploaded_file.name
                st.session_state.batch_processor = BatchProcessor(
                    qa_chain, output_parser, st.session_state.query_parser,
                    st.session_state.rule_engine, st.session_state.confidence_calc, page_mapping,
                )
                st.session_state.rule_engine.extracted_rules = extracted_rules
                st.session_state._file_hash = file_hash
                # Clear chat history when a new document is loaded
                st.session_state.messages = []
                st.session_state.interaction_log = []

            if st.session_state.rule_engine.extracted_rules:
                with st.expander("📋 Detected Policy Rules"):
                    for category, rules in st.session_state.rule_engine.extracted_rules.items():
                        st.markdown(f"**{category.replace('_', ' ').title()}**")
                        if isinstance(rules, dict):
                            for k, v in rules.items():
                                st.write(f"  • {k}: {v}")
                        elif isinstance(rules, list):
                            for r in rules:
                                st.write(f"  • {r}")
                        else:
                            st.write(f"  • {rules}")

        st.divider()
        if "qa_chain" in st.session_state:
            st.success(f"📄 {st.session_state.get('document_name', 'Loaded')}")
            st.success("🤖 Model ready")
        else:
            st.info("Upload a policy PDF to begin")

    # --- Welcome screen (no document uploaded) ---
    if "qa_chain" not in st.session_state:
        st.info("👆 Upload an insurance policy PDF in the sidebar to get started.")
        st.markdown("""
**How it works:**
1. Upload your insurance policy document
2. Ask claim questions in natural language
3. Get AI-powered decisions with justifications and page references

**Example queries:**
- *46-year-old male, knee surgery in Pune, 3-month-old policy*
- *Pre-existing diabetes, 30-year-old female, policy 2 years old*
- *Accidental injury claim, broken arm, 6-month policy*
- *Maternity delivery, policy purchased 10 months ago*
        """)
        return

    # =================================================================
    # Tabs
    # =================================================================
    tab_chat, tab_batch, tab_analytics = st.tabs(
        ["💬 Claims Analysis", "📦 Batch Processing", "📊 Analytics"]
    )

    # --- Tab 1: Chat ---
    with tab_chat:
        # Chat input (always visible at top)
        query = st.chat_input("Describe the insurance claim…")

        if query:
            # Process the new query first so it appears at the top
            st.session_state.messages.insert(0, {"role": "user", "content": query})
            with st.spinner("Analyzing claim…"):
                try:
                    result = st.session_state.batch_processor.process_single_query(query)
                    if "interaction_log" not in st.session_state:
                        st.session_state.interaction_log = []
                    st.session_state.interaction_log.append(result)
                    st.session_state.messages.insert(1, {"role": "assistant", "content": result})
                except Exception as e:
                    st.session_state.messages.insert(1, {
                        "role": "assistant",
                        "content": {"decision": "ERROR", "error": str(e)},
                    })
            st.rerun()

        # Render messages newest-first (they are already stored in reverse order)
        _first_assistant = True
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    render_response(msg["content"], compact=not _first_assistant)
                    _first_assistant = False
                else:
                    st.markdown(msg["content"])

    # --- Tab 2: Batch Processing ---
    with tab_batch:
        st.markdown("Process multiple claims at once — one query per line.")

        batch_input = st.text_area(
            "Claims:",
            height=150,
            placeholder=(
                "46-year-old male, knee surgery, 3-month policy\n"
                "30-year-old female, diabetes, 2-year policy\n"
                "Accidental injury, broken arm, 6-month policy"
            ),
        )

        if st.button("🚀 Process All", type="primary") and batch_input:
            queries = [q.strip() for q in batch_input.split("\n") if q.strip()]
            if not queries:
                st.warning("No valid queries found.")
            else:
                progress = st.progress(0)
                batch_results = []

                for i, q in enumerate(queries):
                    try:
                        result = st.session_state.batch_processor.process_single_query(q)
                        result["query"] = q
                        batch_results.append(result)
                        if "interaction_log" not in st.session_state:
                            st.session_state.interaction_log = []
                        st.session_state.interaction_log.append(result)
                    except Exception as e:
                        batch_results.append({
                            "query": q, "error": str(e), "decision": "ERROR",
                            "amount": 0, "confidence": 0.0, "final_confidence": 0.0,
                            "rule_violations": [], "source_pages": [],
                            "parsed_query": {"completeness_score": 0},
                            "timestamp": datetime.now().isoformat(),
                        })
                    progress.progress((i + 1) / len(queries))

                progress.empty()

                # Summary metrics
                approved = sum(1 for r in batch_results if r.get("decision") == "APPROVED")
                rejected = sum(1 for r in batch_results if r.get("decision") == "REJECTED")
                errors = sum(1 for r in batch_results if r.get("decision") == "ERROR")
                confs = [r.get("final_confidence", r.get("confidence", 0)) for r in batch_results]
                avg_conf = np.mean(confs) if confs else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", len(batch_results))
                c2.metric("Approved", approved)
                c3.metric("Rejected", rejected)
                c4.metric("Errors", errors)

                # Results table
                results_df = pd.DataFrame([
                    {
                        "Query": r.get("query", "")[:80],
                        "Decision": r.get("decision", "Unknown"),
                        "Amount": r.get("amount", 0),
                        "Confidence": f"{r.get('final_confidence', r.get('confidence', 0)):.0%}",
                        "Violations": len(r.get("rule_violations", [])),
                    }
                    for r in batch_results
                ])
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV", csv,
                    f"results_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv",
                )

                # Per-query details
                for i, r in enumerate(batch_results):
                    label = r.get("query", "Query")[:60]
                    with st.expander(f"#{i+1} — {label}"):
                        if "error" in r:
                            st.error(r["error"])
                        else:
                            render_response(r, compact=False)

    # --- Tab 3: Analytics ---
    with tab_analytics:
        create_analytics_dashboard()


if __name__ == "__main__":
    main()