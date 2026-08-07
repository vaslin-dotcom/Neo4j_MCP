EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Each entity MUST be an object with "name", "type", and "description" fields — never a bare string.

Example of correct output shape:
{{
  "entities": [
    {{"name": "Saad Khan", "type": "Person", "description": "Committee Director"}},
    {{"name": "University of Toronto", "type": "Organization", "description": ""}}
  ],
  "relationships": [
    {{"source": "Saad Khan", "relation": "STUDENT_AT", "target": "University of Toronto"}}
  ]
}}

Text:
{chunk}
"""