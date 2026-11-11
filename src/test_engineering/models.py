"""Validated provider output and persisted review contracts."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TestStep(StrictModel):
    step_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    test_data: str = ''
    expected_result: str = Field(min_length=1)


class ManualTestCase(StrictModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    story_id: int = Field(gt=0)
    preconditions: list[str] = Field(default_factory=list)
    steps: list[TestStep] = Field(min_length=1)
    priority: int = Field(ge=1, le=4)
    category: str = Field(min_length=1)
    acceptance_criteria_reference: list[str] = Field(default_factory=list)
    rag_source_references: list[str] = Field(default_factory=list)
    automation_feasibility: Literal['High', 'Medium', 'Low']

    @model_validator(mode='after')
    def sequential_steps(self):
        if [s.step_number for s in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError('Step numbers must be sequential starting at 1')
        return self


class TestDraft(StrictModel):
    test_cases: list[ManualTestCase] = Field(min_length=1)
    gaps: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

