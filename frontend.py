"""
Web application to interact with the RAG system.

Author
------
Nicolas Rojas
"""

import requests
import gradio as gr

API_URL = "http://localhost:8000/query/"


def query_model(text: str) -> str:
    try:
        # Send a POST request to the API
        response = requests.post(API_URL, json={"query": text}, timeout=30)

        # Check whether the request was successful
        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code}"
        result = response.json()["response"]
        result += "\nThis information was obtained from the following files:\n"
        for source in response.json()["source_files"]:
            result += f"- {source}\n"
        return result
    except requests.RequestException as e:
        return f"Error: Could not connect to the API. {str(e)}"


# Create the Gradio interface
interface = gr.Interface(
    fn=query_model,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs="text",
    title="Docuchat",
    description="Enter your question, and I'll provide with an answer from the documentation.",
)

# Launch the app
interface.launch()
