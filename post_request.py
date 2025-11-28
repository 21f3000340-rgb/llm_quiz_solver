import requests

def post_request(url: str, payload: dict):
    """
    Send a JSON POST request and return server response.
    """
    try:
        response = requests.post(url, json=payload, timeout=30)

        try:
            return {"status": response.status_code, "response": response.json()}
        except:
            return {"status": response.status_code, "response": response.text}

    except Exception as e:
        return {"error": str(e)}
