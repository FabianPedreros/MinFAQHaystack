import logging
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Creacion de un almacenamiento de documentos simple y en linea
from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

# Uso de embeddings para la creacion del Retriever
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    scale_score=False,
)

import pandas as pd
df = pd.read_excel("UDEC Preguntas frecuentes.xlsx")

df["Pregunta"] = df["Pregunta"].apply(lambda x: x.strip())

# Embedding de las preguntas, no se realiza embedding de las respuestas, dado que se quiere mapear las preguntas realizadas con las preguntas
questions = list(df["Pregunta"].values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns=({'Pregunta': 'content'}))

docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# Pipeline para realizar preguntas
from haystack.pipelines import FAQPipeline
pipe = FAQPipeline(retriever=retriever)

def responder(entrada):
    prediction = pipe.run(query=entrada, params={"Retriever": {"top_k": 1}})
    answers = prediction["answers"]
    respuesta = ""

    if answers:
        answer = answers[0]
        respuesta = f"{answer.answer}"

    return respuesta

# FastAPI
app = FastAPI()

class PreguntaInput(BaseModel):
    pregunta: str

# ...

@app.post("/responder")
def responder_pregunta(pregunta_input: PreguntaInput):
    pregunta = pregunta_input.pregunta
    respuesta = responder(pregunta)
    return {"respuesta": respuesta}

@app.get("/")
def root():
    return {"message": "¡Bienvenido a mi API!"}

@app.get("/favicon.ico")
def favicon():
    return {"message": "Favicon"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


# uvicorn API_FAQ.py:app --host localhost --port 8000


