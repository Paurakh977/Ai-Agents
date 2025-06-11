import requests

def convert_currency(from_currency: str, to_currency: str, amount: float) -> dict:
    """
    Converts a given amount from one currency to another using a free Open Access API.
    
    Returns:
      {
        "success": bool,
        "from": str,
        "to": str,
        "amount": float,
        "converted": float|null,
        "rate": float|null,
        "error": str|null
      }
    """
    try:
        # Free endpoint – no API key needed, data updates daily :contentReference[oaicite:0]{index=0}
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("result") != "success":
            raise Exception(data.get("error-type", "API returned error"))

        rates = data.get("rates", {})
        rate = rates.get(to_currency.upper())
        
        if rate is None:
            raise Exception(f"No rate from {from_currency} to {to_currency}")
        
        converted = amount * rate

        return {
            "success": True,
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "amount": amount,
            "converted": converted,
            "rate": rate,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "amount": amount,
            "converted": None,
            "rate": None,
            "error": str(e)
        }
