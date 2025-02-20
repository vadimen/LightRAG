from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "category"]

PROMPTS["entity_extraction"] = """Given text and entity types, identify entities and relationships.
Use {language} as output language.

1. For each entity:
- name: Name in input language
- type: [{entity_types}]
- desc: Brief description
Format: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<desc>)

2. For related entities, extract:
- source/target names
- brief relationship desc
- strength (1-10)
- keywords
Format: ("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<desc>{tuple_delimiter}<keywords>{tuple_delimiter}<strength>)

3. Add document keywords as: ("content_keywords"{tuple_delimiter}<keywords>)

Use **{record_delimiter}** as delimiter. End with {completion_delimiter}

{examples}

Entity_types: {entity_types}
Text: {input_text}
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example:
Entity_types: [person, technology]
Text: Alex used the quantum computer while Bob supervised.
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Quantum systems operator"){record_delimiter}
("entity"{tuple_delimiter}"Bob"{tuple_delimiter}"person"{tuple_delimiter}"Project tech supervisor"){record_delimiter}
("entity"{tuple_delimiter}"Quantum Computer"{tuple_delimiter}"technology"{tuple_delimiter}"Advanced quantum processing system"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Quantum Computer"{tuple_delimiter}"Uses"{tuple_delimiter}"operation"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Bob"{tuple_delimiter}"Alex"{tuple_delimiter}"Supervises"{tuple_delimiter}"management"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"quantum computing, supervision"){completion_delimiter}"""
]

PROMPTS["summarize_entity_descriptions"] = """Combine descriptions of entities into single summary in {language}.

Entities: {entity_name}
Description List: {description_list}
Output:"""

PROMPTS["entiti_continue_extraction"] = """Add missed entities using same format:"""

PROMPTS["entiti_if_loop_extraction"] = """Answer YES | NO if entities still need to be added."""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS["rag_response"] = """Generate concise response based on Knowledge Base.
Consider timestamps for conflicting info.
History: {history}
Knowledge Base: {context_data}

Rules:
- Format: {response_type}
- Use markdown
- Match user language
- Only use provided info
- Say if unknown"""

PROMPTS["keywords_extraction"] = """Extract high/low-level keywords from query and history.
History: {history}
Query: {query}

Output JSON with:
- high_level_keywords: concepts/themes
- low_level_keywords: specific details

{examples}"""

PROMPTS["keywords_extraction_examples"] = [
    """Example:
Query: "Impact of AI on jobs?"
Output:
{
  "high_level_keywords": ["AI impact", "Employment"],
  "low_level_keywords": ["Automation", "Job displacement"]
}"""
]

PROMPTS["naive_rag_response"] = """Generate response based on Document Chunks and history.
History: {history}
Document Chunks: {content_data}

Rules:
- Format: {response_type}
- Use markdown
- Match user language
- Only use provided info
- Say if unknown"""

PROMPTS["similarity_check"] = """Compare semantic similarity of questions:
Q1: {original_prompt}
Q2: {cached_prompt}

Return 0-1 score:
0: Unrelated/different topic/conditions 
0.5: Partially related
1: Identical"""

PROMPTS["mix_rag_response"] = """Generate response using Knowledge Graph and Document Chunks.
History: {history}

Sources:
1. KG: {kg_context}
2. DC: {vector_context}

Rules:
- Format: {response_type}
- Use markdown with sections
- Match user language 
- Only use provided data
- Include [KG/DC] references
- Say if unknown"""
