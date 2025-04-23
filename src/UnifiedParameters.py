from enum import Enum
from src.base.DoubleQuantumDot import DQDAttributes
from src.base.AttributeInterpreter import NoAttributeParameters


# Combine DQDAttributes and NoAttributeParameters into a single dictionary
def _combine_enums():
    combined = {}
    for attr in DQDAttributes:
        combined[attr.name] = attr.value
    for param in NoAttributeParameters:
        combined[param.name] = param.value
    return combined


# Create the UnifiedParameters class dynamically
UnifiedParameters = Enum("UnifiedParameters", _combine_enums())