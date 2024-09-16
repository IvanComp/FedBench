# Usa un'immagine base di Python
FROM python:3.9

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file requirements.txt
COPY requirements.txt ./

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dell'applicazione
COPY . .

# Comando per avviare il client
CMD ["python", "client.py"]
