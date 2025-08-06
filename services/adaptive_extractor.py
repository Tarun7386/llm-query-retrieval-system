# services/adaptive_extractor.py
import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

class AdaptiveAnswerExtractor:
    """Universal answer extractor that adapts to any document type and domain."""
    
    def __init__(self):
        self.document_patterns = {}
        self.domain_vocabulary = defaultdict(set)
        
    def analyze_document(self, chunks: List[str], doc_id: str) -> None:
        """Analyze any document type to discover patterns and terminology."""
        
        full_text = " ".join(chunks).lower()
        
        # 1. Universal numerical patterns
        numerical_patterns = self._extract_numerical_patterns(full_text)
        
        # 2. Domain vocabulary discovery
        domain_vocab = self._discover_domain_vocabulary(full_text)
        
        # 3. Document structure patterns
        structure_patterns = self._analyze_document_structure(full_text)
        
        # 4. Relationship patterns (cause-effect, condition-result)
        relationship_patterns = self._extract_relationship_patterns(full_text)
        
        # 5. Entity recognition
        entities = self._extract_entities(full_text)
        
        # Store all patterns for this document
        self.document_patterns[doc_id] = {
            'numerical': numerical_patterns,
            'vocabulary': domain_vocab,
            'structure': structure_patterns,
            'relationships': relationship_patterns,
            'entities': entities,
            'text_length': len(full_text),
            'sentence_count': len([s for s in full_text.split('.') if len(s.strip()) > 10])
        }
    
    def _extract_numerical_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract all types of numerical information."""
        
        patterns = {
            'percentages': re.findall(r'(\d+(?:\.\d+)?)\s*%', text),
            'currency': re.findall(r'(?:\$|€|£|₹|inr|usd|eur|gbp|rs\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)', text),
            'dates': re.findall(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})', text),
            'time_periods': {
                'days': re.findall(r'(\d+)\s*days?', text),
                'weeks': re.findall(r'(\d+)\s*weeks?', text),
                'months': re.findall(r'(\d+)\s*months?', text),
                'years': re.findall(r'(\d+)\s*years?', text),
                'hours': re.findall(r'(\d+)\s*hours?', text),
                'minutes': re.findall(r'(\d+)\s*minutes?', text)
            },
            'measurements': re.findall(r'(\d+(?:\.\d+)?)\s*(kg|lb|cm|inch|meter|mile|km|ft)', text),
            'versions': re.findall(r'(?:version|v\.?)\s*(\d+(?:\.\d+)*)', text),
            'quantities': re.findall(r'(\d+(?:,\d+)*)\s*(?:items?|pieces?|units?|count)', text)
        }
        
        return patterns
    
    def _discover_domain_vocabulary(self, text: str) -> Dict[str, Set[str]]:
        """Automatically discover domain-specific vocabulary."""
        
        # Common domain indicators
        domain_indicators = {
            'action_terms': ['process', 'procedure', 'method', 'approach', 'technique', 'operation'],
            'requirement_terms': ['must', 'shall', 'required', 'mandatory', 'necessary', 'essential'],
            'condition_terms': ['if', 'when', 'unless', 'provided that', 'subject to', 'in case of'],
            'negative_terms': ['not', 'cannot', 'prohibited', 'forbidden', 'excluded', 'denied'],
            'positive_terms': ['allowed', 'permitted', 'included', 'covered', 'eligible', 'entitled'],
            'temporal_terms': ['before', 'after', 'during', 'within', 'until', 'from', 'since'],
            'comparison_terms': ['more than', 'less than', 'equal to', 'minimum', 'maximum', 'at least'],
            'causation_terms': ['because', 'due to', 'as a result', 'therefore', 'consequently', 'leads to']
        }
        
        found_terms = defaultdict(set)
        
        # Find domain-specific terms
        for category, terms in domain_indicators.items():
            for term in terms:
                if term in text:
                    # Find variations and context
                    pattern = r'\b' + re.escape(term) + r'\w*\b'
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    found_terms[category].update(matches)
        
        # Extract frequent noun phrases (potential domain terms)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrase_counts = Counter(noun_phrases)
        frequent_phrases = {phrase for phrase, count in phrase_counts.items() if count >= 3}
        found_terms['domain_specific'] = frequent_phrases
        
        return dict(found_terms)
    
    def _analyze_document_structure(self, text: str) -> Dict[str, List[str]]:
        """Analyze document structure patterns."""
        
        structure = {
            'headings': [],
            'lists': [],
            'definitions': [],
            'procedures': []
        }
        
        # Detect headings (various formats)
        heading_patterns = [
            r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS HEADINGS
            r'^\d+\.\s*[A-Z][^.]*$',     # 1. Numbered headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',  # Title Case Headings:
            r'^\([a-z]\)\s*[A-Z]',       # (a) Lettered items
            r'^\*\s*[A-Z]'               # * Bullet points
        ]
        
        for pattern in heading_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            structure['headings'].extend(matches)
        
        # Detect list items
        list_patterns = [
            r'^\s*[-•]\s*(.+)$',         # Bullet lists
            r'^\s*\d+\.\s*(.+)$',        # Numbered lists
            r'^\s*[a-z]\)\s*(.+)$'       # Lettered lists
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            structure['lists'].extend(matches)
        
        # Detect definitions
        definition_patterns = [
            r'([A-Z][a-z\s]+)\s*means\s*(.+?)(?:\.|$)',
            r'([A-Z][a-z\s]+)\s*is\s*defined\s*as\s*(.+?)(?:\.|$)',
            r'([A-Z][a-z\s]+):\s*(.+?)(?:\.|$)'
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            structure['definitions'].extend([f"{term}: {definition}" for term, definition in matches])
        
        return structure
    
    def _extract_relationship_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract cause-effect and conditional relationships."""
        
        relationships = {
            'cause_effect': [],
            'conditions': [],
            'requirements': [],
            'exceptions': []
        }
        
        # Cause-effect patterns
        cause_effect_patterns = [
            r'(.+?)\s*(?:results? in|leads? to|causes?)\s*(.+?)(?:\.|$)',
            r'(?:because|due to|as a result of)\s*(.+?),?\s*(.+?)(?:\.|$)',
            r'(.+?)\s*therefore\s*(.+?)(?:\.|$)'
        ]
        
        for pattern in cause_effect_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relationships['cause_effect'].extend([f"{cause} → {effect}" for cause, effect in matches])
        
        # Conditional patterns
        condition_patterns = [
            r'(?:if|when|unless)\s*(.+?),?\s*then\s*(.+?)(?:\.|$)',
            r'provided that\s*(.+?),?\s*(.+?)(?:\.|$)',
            r'in case of\s*(.+?),?\s*(.+?)(?:\.|$)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            relationships['conditions'].extend([f"IF {condition} THEN {result}" for condition, result in matches])
        
        return relationships
    
    def _extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract different types of entities."""
        
        entities = {
            'organizations': set(),
            'locations': set(),
            'products': set(),
            'people': set(),
            'technologies': set()
        }
        
        # Organization patterns
        org_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Ltd|Corp|Company|Corporation|LLC|Organization)\b',
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].update(matches)
        
        # Technology/Product patterns
        tech_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:System|Software|Platform|Application|Tool|API)\b',
            r'\b(?:version|v\.?)\s*\d+(?:\.\d+)*\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['technologies'].update(matches)
        
        return {k: v for k, v in entities.items() if v}  # Remove empty sets
    
    def extract_answer(self, chunk_content: str, question: str, doc_id: str) -> str:
        """Universal answer extraction that works with any document type."""
        
        content = self._clean_content(chunk_content)
        question_lower = question.lower()
        
        # Get document-specific patterns
        doc_patterns = self.document_patterns.get(doc_id, {})
        
        # Multi-strategy extraction with confidence scoring
        extraction_strategies = [
            ('semantic', self._semantic_extract_universal),
            ('numerical', self._numerical_extract),
            ('structural', self._structural_extract),
            ('contextual', self._contextual_extract),
            ('keyword', self._keyword_extract)
        ]
        
        best_answer = None
        best_confidence = 0
        
        for strategy_name, strategy_func in extraction_strategies:
            try:
                answer = strategy_func(content, question_lower, doc_patterns)
                if answer:
                    confidence = self._calculate_answer_confidence(answer, question, content)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_answer = answer
            except Exception:
                continue  # Skip failed strategies
        
        if best_answer and best_confidence > 0.3:
            return best_answer
        
        # Fallback to generic extraction
        return self._generic_extract_enhanced(content, question_lower)
    
    def _semantic_extract_universal(self, content: str, question: str, doc_patterns: Dict) -> Optional[str]:
        """Universal semantic extraction for any domain."""
        
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
        question_type = self._classify_question_universal(question)
        
        extraction_methods = {
            'what': self._extract_definition_universal,
            'how': self._extract_procedure_universal,
            'when': self._extract_temporal_universal,
            'where': self._extract_location_universal,
            'why': self._extract_causation_universal,
            'who': self._extract_agent_universal,
            'which': self._extract_selection_universal,
            'yes_no': self._extract_boolean_universal,
            'quantity': self._extract_numerical_universal
        }
        
        method = extraction_methods.get(question_type, self._extract_general_universal)
        return method(sentences, question, doc_patterns)
    
    def _classify_question_universal(self, question: str) -> str:
        """Universal question classification."""
        
        question_words = question.lower().split()
        
        # WH-questions
        if any(word.startswith('what') for word in question_words):
            return 'what'
        elif any(word.startswith('how') for word in question_words):
            return 'how'
        elif any(word.startswith('when') for word in question_words):
            return 'when'
        elif any(word.startswith('where') for word in question_words):
            return 'where'
        elif any(word.startswith('why') for word in question_words):
            return 'why'
        elif any(word.startswith('who') for word in question_words):
            return 'who'
        elif any(word.startswith('which') for word in question_words):
            return 'which'
        
        # Yes/No questions
        elif any(word in ['is', 'are', 'does', 'do', 'can', 'will', 'would', 'should'] for word in question_words[:2]):
            return 'yes_no'
        
        # Quantity questions
        elif any(word in ['how many', 'how much', 'number'] for word in ' '.join(question_words[:3]).split()):
            return 'quantity'
        
        return 'general'
    
    def _extract_definition_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract definitions for any domain."""
        
        # Definition indicators
        definition_patterns = [
            r'(.+?)\s*(?:means?|is|are|refers? to|defined as|denotes?)\s*(.+)',
            r'(.+?):\s*(.+)',  # Colon definitions
            r'(.+?)\s*-\s*(.+)'  # Dash definitions
        ]
        
        question_keywords = set(question.lower().split())
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Check relevance
            if len(question_keywords.intersection(sentence_words)) >= 1:
                for pattern in definition_patterns:
                    match = re.search(pattern, sentence_lower)
                    if match:
                        term, definition = match.groups()
                        if any(keyword in term or keyword in definition for keyword in question_keywords):
                            return f"{term.strip().title()} {definition.strip()}."
        
        return None
    
    def _extract_numerical_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract numerical answers for any domain."""
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for numbers in relevant sentences
            if any(keyword in sentence_lower for keyword in question.lower().split()[:3]):
                # Find various numerical patterns
                numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', sentence)
                percentages = re.findall(r'\d+(?:\.\d+)?%', sentence)
                currencies = re.findall(r'[\$€£₹]\d+(?:,\d+)*(?:\.\d+)?', sentence)
                
                if numbers or percentages or currencies:
                    return sentence.strip() + "."
        
        return None
    
    def _extract_boolean_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract yes/no answers for any domain."""
        
        positive_indicators = ['yes', 'true', 'correct', 'allowed', 'permitted', 'can', 'will', 'does', 'is', 'are']
        negative_indicators = ['no', 'false', 'incorrect', 'not allowed', 'prohibited', 'cannot', 'will not', 'does not', 'is not', 'are not']
        
        question_keywords = set(question.lower().split())
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            # Check relevance
            if len(question_keywords.intersection(sentence_words)) >= 1:
                # Check for positive indicators
                if any(indicator in sentence_lower for indicator in positive_indicators):
                    return f"Yes, {sentence.strip().lower()}."
                
                # Check for negative indicators
                if any(indicator in sentence_lower for indicator in negative_indicators):
                    return f"No, {sentence.strip().lower()}."
        
        return None
    
    def _extract_procedure_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract procedural information for any domain."""
        
        procedure_indicators = ['step', 'process', 'method', 'procedure', 'way', 'manner', 'approach']
        sequence_indicators = ['first', 'second', 'then', 'next', 'finally', 'subsequently']
        
        best_sentence = ""
        max_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Score based on procedure indicators
            procedure_score = sum(1 for indicator in procedure_indicators if indicator in sentence_lower)
            sequence_score = sum(1 for indicator in sequence_indicators if indicator in sentence_lower)
            
            # Question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            relevance_score = len(question_words.intersection(sentence_words))
            
            total_score = procedure_score + sequence_score + relevance_score
            
            if total_score > max_score and len(sentence) > 20:
                max_score = total_score
                best_sentence = sentence
        
        return best_sentence.strip() + "." if best_sentence else None
    
    def _extract_temporal_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract temporal information for any domain."""
        
        temporal_patterns = [
            r'\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?|seconds?)',
            r'(?:before|after|during|within|until|from|since)\s+\w+',
            r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            
            if len(question_words.intersection(sentence_words)) >= 1:
                # Check for temporal patterns
                if any(re.search(pattern, sentence_lower) for pattern in temporal_patterns):
                    return sentence.strip() + "."
        
        return None
    
    def _extract_causation_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract causal relationships for any domain."""
        
        causation_indicators = ['because', 'due to', 'as a result', 'therefore', 'consequently', 'leads to', 'causes', 'results in']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            
            if len(question_words.intersection(sentence_words)) >= 1:
                # Check for causation indicators
                if any(indicator in sentence_lower for indicator in causation_indicators):
                    return sentence.strip() + "."
        
        return None
    
    def _extract_general_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """General extraction for any type of question."""
        
        question_words = set(question.lower().split())
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap_score = len(question_words.intersection(sentence_words))
            
            if overlap_score > 0:
                scored_sentences.append((sentence, overlap_score))
        
        if scored_sentences:
            best_sentence = max(scored_sentences, key=lambda x: x[1])[0]
            return best_sentence.strip() + "."
        
        return None
    
    def _numerical_extract(self, content: str, question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract numerical information."""
        return self._extract_numerical_universal(content.split('.'), question, doc_patterns)
    
    def _structural_extract(self, content: str, question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract based on document structure."""
        
        # Look for structured information (lists, headings, etc.)
        lines = content.split('\n')
        question_keywords = set(question.lower().split())
        
        for line in lines:
            line_words = set(line.lower().split())
            if len(question_keywords.intersection(line_words)) >= 2 and len(line.strip()) > 20:
                return line.strip() + "."
        
        return None
    
    def _contextual_extract(self, content: str, question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract based on context and relationships."""
        
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
        question_keywords = set(question.lower().split())
        
        # Look for sentences with high contextual relevance
        for i, sentence in enumerate(sentences):
            sentence_words = set(sentence.lower().split())
            
            # Current sentence relevance
            current_relevance = len(question_keywords.intersection(sentence_words))
            
            # Context from surrounding sentences
            context_relevance = 0
            if i > 0:  # Previous sentence
                prev_words = set(sentences[i-1].lower().split())
                context_relevance += len(question_keywords.intersection(prev_words)) * 0.5
            
            if i < len(sentences) - 1:  # Next sentence
                next_words = set(sentences[i+1].lower().split())
                context_relevance += len(question_keywords.intersection(next_words)) * 0.5
            
            total_relevance = current_relevance + context_relevance
            
            if total_relevance >= 2:
                return sentence.strip() + "."
        
        return None
    
    def _keyword_extract(self, content: str, question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract based on keyword matching."""
        
        sentences = content.split('.')
        question_keywords = question.lower().split()[:5]  # Use first 5 words
        
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for keyword in question_keywords if keyword in sentence_lower)
            
            if matches > max_matches and len(sentence.strip()) > 20:
                max_matches = matches
                best_sentence = sentence
        
        return best_sentence.strip() + "." if best_sentence and max_matches >= 2 else None
    
    def _calculate_answer_confidence(self, answer: str, question: str, content: str) -> float:
        """Calculate confidence score for an extracted answer."""
        
        score = 0.0
        
        # Word overlap between question and answer
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap_ratio = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0
        score += overlap_ratio * 0.4
        
        # Answer length appropriateness
        if 20 <= len(answer) <= 200:
            score += 0.3
        elif len(answer) < 10 or len(answer) > 400:
            score -= 0.2
        
        # Contains specific information (numbers, dates, etc.)
        if re.search(r'\d+', answer):
            score += 0.2
        
        # Grammatical completeness
        if answer.strip().endswith('.') and answer.strip()[0].isupper():
            score += 0.1
        
        # Presence of definitive language
        definitive_terms = ['is', 'are', 'will', 'must', 'shall', 'can', 'cannot', 'yes', 'no']
        if any(term in answer.lower() for term in definitive_terms):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _generic_extract_enhanced(self, content: str, question: str) -> str:
        """Enhanced generic extraction as final fallback."""
        
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
        
        if not sentences:
            return "No relevant information found."
        
        # Simple keyword matching
        question_keywords = set(question.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_keywords.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            best_sentence = max(scored_sentences, key=lambda x: x[1])[0]
            return best_sentence.strip() + "."
        
        # Return first meaningful sentence if no overlap
        return sentences[0].strip() + "."
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        content = re.sub(r'--- Page \d+ ---', '', content)
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\%\$€£₹]', ' ', content)
        return content.strip()
    
    # Additional helper methods for specific extraction types
    def _extract_agent_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract information about who/which entity."""
        
        agent_indicators = ['responsible', 'manages', 'handles', 'oversees', 'performs', 'executes']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            
            if len(question_words.intersection(sentence_words)) >= 1:
                # Check for agent indicators
                if any(indicator in sentence_lower for indicator in agent_indicators):
                    return sentence.strip() + "."
        
        return None
    
    def _extract_location_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract location information."""
        
        location_patterns = [
            r'\b(?:in|at|on|from|to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # City, State
            r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            
            if len(question_words.intersection(sentence_words)) >= 1:
                # Check for location patterns
                if any(re.search(pattern, sentence) for pattern in location_patterns):
                    return sentence.strip() + "."
        
        return None
    
    def _extract_selection_universal(self, sentences: List[str], question: str, doc_patterns: Dict) -> Optional[str]:
        """Extract selection/choice information."""
        
        selection_indicators = ['option', 'choice', 'alternative', 'either', 'or', 'between', 'among']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check question relevance
            question_words = set(question.lower().split())
            sentence_words = set(sentence_lower.split())
            
            if len(question_words.intersection(sentence_words)) >= 1:
                # Check for selection indicators
                if any(indicator in sentence_lower for indicator in selection_indicators):
                    return sentence.strip() + "."
        
        return None
