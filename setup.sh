pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
python database_handler.py -c
ollama pull mixtral
