# Complete patch for gradio_client/utils.py

def get_desc(schema):
    """Get a description from a JSON schema if available."""
    description = schema.get("description", "")
    if description:
        return f"  # {description}"
    return ""

def get_type(schema):
    """Get the type of a JSON schema."""
    # Handle non-dict schema
    if not isinstance(schema, dict):
        if isinstance(schema, bool):
            return "bool"
        return str(type(schema).__name__)
    
    if "type" in schema:
        return schema["type"]
    
    for t in ["oneOf", "anyOf", "object", "array"]:
        if t in schema:
            return t
    
    if "additionalProperties" in schema:
        return "additionalProperties"
    
    return "any"

def _json_schema_to_python_type(schema, defs=None):
    """Convert a JSON schema to a Python type."""
    defs = defs or {}
    if schema is None:
        return "None"
    
    # Check if schema is a dictionary before trying to use 'in' operator
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _json_schema_to_python_type(defs[schema["$ref"].split("/")[-1]], defs)
        
        if "const" in schema:
            return repr(schema["const"])
        
        if "enum" in schema:
            return " | ".join([repr(i) for i in schema["enum"]])
    
    # Handle boolean schema (this was the cause of the original error)
    if isinstance(schema, bool):
        return "bool"
    
    # Get type based on schema
    type_ = get_type(schema)
    
    # Handle different types
    if type_ == "object":
        des = [
            f"{n}: {_json_schema_to_python_type(v, defs)}{get_desc(v)}"
            for n, v in schema.get("properties", {}).items()
        ]
        return f"dict[{', '.join(des)}]" if des else "dict"
    
    if type_ == "array":
        items = schema.get("items", {})
        elements = _json_schema_to_python_type(items, defs)
        return f"list[{elements}]"
    
    if type_ == "anyOf" or type_ == "oneOf":
        desc = " | ".join([_json_schema_to_python_type(i, defs) for i in schema[type_]])
        return desc
    
    if type_ == "string":
        if "format" in schema and schema["format"] == "binary":
            return "bytes"
        return "str"
    
    if type_ == "additionalProperties" and isinstance(schema.get("additionalProperties"), dict):
        return f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"
    
    return type_

# This is the main function called by gradio
def json_schema_to_python_type(schema):
    """Convert a JSON schema to a Python type representation."""
    if not isinstance(schema, dict):
        return str(type(schema).__name__)
    type_ = _json_schema_to_python_type(schema, schema.get("$defs"))
    return type_