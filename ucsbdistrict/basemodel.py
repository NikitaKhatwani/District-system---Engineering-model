from pydantic import UUID4, BaseModel, Field
import uuid

class BaseEquipment(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4, description="The unique id of the equipment")
