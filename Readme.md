* Iris FastAPI – ML Model as API

a)Opis projektu

Projekt przedstawia prosty system sztucznej inteligencji udostępniony jako usługa webowa (API).  
System wykorzystuje wcześniej wytrenowany model uczenia maszynowego do klasyfikacji danych ze zbioru Iris.

Aplikacja przyjmuje dane wejściowe w formacie JSON i zwraca wynik predykcji modelu.

Projekt łączy elementy:
- uczenia maszynowego (Python, scikit-learn),
- aplikacji webowej (FastAPI),
- pracy zespołowej z użyciem systemu kontroli wersji Git,
- zarządzania środowiskiem przy użyciu narzędzia 'uv'.


b)Wymagania

- Python 3.10+
- uv 



c)Instalacja i przygotowanie środowiska

1. Sklonowanie repozytorium:

bash
git clone https://github.com/Ania2000/iris-fastapi.git

cd iris-fastapi

2. Utworzenie środowiska i instalacja zależności

uv venv uv sync 

3. Trenowanie modelu

Model trenowany jest lokalnie i zapisywany do pliku:

models/iris_model.joblib 

Aby wytrenować model:

uv run python scripts/train.py 

Po zakończeniu treningu w folderze models/ pojawi się zapisany model.

4. Uruchomienie aplikacji

Aby uruchomić serwer FastAPI:

uvicorn main:app --reload --host 127.0.0.1 --port 8000  

Aplikacja będzie dostępna pod adresem:

http://127.0.0.1:8000 

Automatyczna dokumentacja API:

http://127.0.0.1:8000/docs 

5. Endpointy API

GET /health

Endpoint sprawdzający poprawność działania aplikacji.

Przykładowa odpowiedź:

{ "status": "ok" } 

POST /predict

Endpoint wykonujący predykcję modelu.

Dane wejściowe (JSON)

{
  "sepal_length": 6.2,
  "sepal_width": 2.8,
  "petal_length": 4.8,
  "petal_width": 1.8
}
Odpowiedź

{
  "predicted_class": 1,
  "predicted_label": "versicolor"
}

Użyty model

W projekcie wykorzystano model klasyfikacyjny z biblioteki scikit-learn:

zbiór danych: Iris Dataset,

typ zadania: klasyfikacja,

model trenowany lokalnie,

model zapisany przy użyciu biblioteki joblib,

model ładowany podczas uruchamiania aplikacji (nie jest trenowany przy każdym starcie serwera).