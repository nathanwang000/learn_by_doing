import json
from typing import Dict

def pretty_format_dict(d: Dict)->str:
    return json.dumps(d, indent=4)