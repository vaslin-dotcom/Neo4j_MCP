EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Each entity must have: name, type, description.
Each relationship must have: source (a full entity object), relation, target (a full entity object).
source and target are nested objects with their own name, type, and description - not plain strings.

Only extract entities that represent real-world people, organizations, places, groups, or named
things. Do NOT extract numeric values, ratings, scores, or years as standalone entities unless
the text is specifically discussing a historical year as an event.

Example of correct output shape:
{{
  "entities": [
    {{"name": "Saad Khan", "type": "Person", "description": "Committee Director"}},
    {{"name": "University of Toronto", "type": "Organization", "description": "Educational institution"}}
  ],
  "relationships": [
    {{
      "source": {{"name": "Saad Khan", "type": "Person", "description": "Committee Director"}},
      "relation": "STUDENT_AT",
      "target": {{"name": "University of Toronto", "type": "Organization", "description": "Educational institution"}}
    }}
  ]
}}

Text:
{chunk}
"""