EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Each entity MUST be an object with "name", "type", and "description" fields — never a bare string.
This applies everywhere an entity appears, including as a "source" or "target" inside a relationship.

Example of correct output shape:
{{
  "entities": [
    {{"name": "Saad Khan", "type": "Person", "description": "Committee Director"}},
    {{"name": "University of Toronto", "type": "Organization", "description": ""}}
  ],
  "relationships": [
    {{
      "source": {{"name": "Saad Khan", "type": "Person", "description": "Committee Director"}},
      "relation": "STUDENT_AT",
      "target": {{"name": "University of Toronto", "type": "Organization", "description": ""}}
    }}
  ]
}}
Text:
{chunk}
"""