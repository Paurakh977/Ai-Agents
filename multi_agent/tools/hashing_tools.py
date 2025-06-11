import base64

def encode_string(s: str) -> dict:
    """
    Reversibly “encodes” a string using Base64.
    Returns: {"success": bool, "encoded": str or None, "error": str or None}
    """
    try:
        encoded = base64.b64encode(s.encode("utf-8")).decode("utf-8")
        return {"success": True, "encoded": encoded, "error": None}
    except Exception as e:
        return {"success": False, "encoded": None, "error": str(e)}

def decode_string(encoded: str) -> dict:
    """
    Decodes a Base64‐encoded string produced by encode_string().
    Returns: {"success": bool, "decoded": str or None, "error": str or None}
    """
    try:
        decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
        return {"success": True, "decoded": decoded, "error": None}
    except Exception as e:
        return {"success": False, "decoded": None, "error": str(e)}

def reverse_string(s: str) -> dict:
    """
    Returns the reverse of the input string.
    Returns: {"success": True, "reversed": str}
    """
    try:
        rev = s[::-1]
        return {"success": True, "reversed": rev, "error": None}
    except Exception as e:
        return {"success": False, "reversed": None, "error": str(e)}
