import random
import requests

def get_joke(category: str) -> dict:
    """
    Returns a joke of the given category.
    Supported categories: "programming", "ai", "dad"
    
    Returns:
      {
        "success": bool,
        "type": str,      # the requested category
        "joke": str|null, # the joke text
        "error": str|null # error message if any
      }
    """
    cat = category.lower()
    
    try:
        if cat == "programming":
            # Use JokeAPI for programming jokes
            resp = requests.get(
                "https://v2.jokeapi.dev/joke/Programming",
                params={"type": "single"},
                headers={"Accept": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                raise Exception(data.get("message", "Unknown error from JokeAPI"))
            joke = data.get("joke")
        
        elif cat == "ai":
            # Curated AI jokes
            ai_jokes = [
                "Why did the AI go on a diet? It had too many bytes!",
                "What do you call a lazy AI? Artificially unintelligent.",
                "Why don't AI ever tell secrets? Because they're afraid of data leaks!"
            ]
            joke = random.choice(ai_jokes)
        
        elif cat == "dad":
            # Fetch from icanhazdadjoke.com
            resp = requests.get(
                "https://icanhazdadjoke.com/",
                headers={
                    "Accept": "application/json",
                    "User-Agent": "YourAgentName (contact@example.com)"
                }
            )
            resp.raise_for_status()
            joke = resp.json().get("joke")
        
        else:
            return {
                "success": False,
                "type": category,
                "joke": None,
                "error": f"Unsupported category '{category}'. Choose from programming, ai, dad."
            }

        return {
            "success": True,
            "type": cat,
            "joke": joke,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "type": category,
            "joke": None,
            "error": str(e)
        }




