EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY valid JSON matching this exact schema, no other text:

{schema}

Text:
{chunk}
"""