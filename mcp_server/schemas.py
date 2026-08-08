from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(description="The entity's name, exactly as it appears in the text")
    type: str = Field(description="The entity's category, e.g. Person, Organization, Location, Committee")
    description: str = Field(default="", description="A brief description of this entity")


class Relationship(BaseModel):
    source: Entity = Field(description="The source entity")
    relation: str = Field(description="The relationship label connecting source to target, e.g. WORKS_AT, PART_OF - use the field name 'relation', NOT 'type'")
    target: Entity = Field(description="The target entity")


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default=[], description="All entities found in the text")
    relationships: list[Relationship] = Field(default=[], description="All relationships between entities found in the text - each MUST have 'source', 'relation', 'target' fields")