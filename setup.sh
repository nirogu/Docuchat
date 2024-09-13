pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama
python database_handler.py -c
