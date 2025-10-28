# 1. Usa un'immagine Python ufficiale e leggera
FROM python:3.10-slim

# 2. Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# 3. Copia *solo* il file dei requisiti
COPY requirements.txt requirements.txt

# 4. Installa le librerie Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia tutto il resto del tuo codice (app.py, ecc.)
COPY . .

# 6. Esponi la porta di default di Streamlit
EXPOSE 8501

# 7. Comando per avviare l'app
CMD streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.headless=true --server.enableCORS=false