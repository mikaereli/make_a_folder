from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Инициализация лемматизатора и списка стоп-слов
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class FolderRequest(BaseModel):
    query: str

@app.post("/create_folder")
async def create_folder(request: FolderRequest):
    query = request.query
    
    # Обработка запроса
    tokens = word_tokenize(query.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    
    # Выбор названия папки (используем первые два значимых слова)
    folder_name = '_'.join(tokens[:2])
    
    # Создание папки
    try:
        os.makedirs(folder_name, exist_ok=True)
        return {"message": f"Folder '{folder_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)