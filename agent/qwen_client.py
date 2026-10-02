import base64
import json
import requests


SERVER_URL = "hidden"
#indirizzo server locale

MODEL_NAME = "hidden"
#nome del modello caricato da llama-server


def image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    #leggiamo i byte dell'immagine

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    #convertiamo i byte in base64

    return f"data:image/png;base64,{image_base64}"
    #costruiamo la data url che possiamo inviare al server


def ask_qwen(image_path, question, memory=None):
    image_url = image_to_data_url(image_path)
    #trasformiamo l'immagine in una data url

    if memory is None:
        memory = {
            "concepts": {},
            "relations": {}
        }
    #se non passiamo memoria usiamo una memoria vuota

    if "concepts" not in memory:
        memory = {
            "concepts": memory,
            "relations": {}
        }
    #manteniamo compatibilita con il vecchio formato della memoria

    memory_text = json.dumps(
        memory,
        indent=2,
        ensure_ascii=False
    )
    #trasformiamo la memoria in testo json

    prompt = f"""
You are solving a visual classification task.

Question:
{question}

The following external knowledge has been retrieved from memory:

{memory_text}

Use this knowledge as additional conceptual information when reasoning
about the image.

Do not assume that a concept is present in the image just because it
appears in memory. You must inspect the image.

For "used_concepts":
- return only concepts that are present in the provided memory
- include a concept if information from its definition or visual cues
  was used to make the classification
- return an empty list if no memory concept was used
- do not invent concepts

For "used_relations":
- return only relations that are present in the provided memory
- include a relation only if it was actually used
- return an empty list if no memory relation was used

Return JSON only using exactly this format:

{{
    "answer": 0,
    "used_concepts": [],
    "used_relations": [],
    "evidence": "short visual evidence"
}}

The answer must be either 0 or 1.
"""
    #costruiamo il prompt che contiene domanda, immagine e memoria

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 150,
        "response_format": {
            "type": "json_object"
        }
    }
    #prepariamo la richiesta da mandare a qwen

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    #inviamo la richiesta a llama-server

    response.raise_for_status()
    #se il server restituisce un errore http solleviamo un'eccezione

    result = response.json()
    #trasformiamo la risposta del server in un dizionario python

    return result["choices"][0]["message"]["content"]
    #restituiamo solo il contenuto generato da qwen


def extract_concepts(question):
    prompt = f"""
You are extracting the minimal symbolic representation of a visual task.

Task:
{question}

Your job is NOT to solve the task.

Extract ONLY the visual concepts and semantic relations that are explicitly
required by the task.

IMPORTANT RULES:

1. Do not invent concepts that are not required by the task.
2. Do not add generic concepts such as "object", "shape", "background".
3. Do not add alternative shapes that are not mentioned.
4. Include a concept only if the task refers to it.
5. Include a relation only if the task explicitly requires a relation.
6. Return the MINIMAL set of concepts and relations necessary to represent
   the task.
7. Do not describe the image.
8. Do not answer the task.

Return JSON only using exactly this format:

{{
    "concepts": [],
    "relations": []
}}
"""
    #costruiamo il prompt per estrarre i concetti dalla domanda

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 100,
        "response_format": {
            "type": "json_object"
        }
    }
    #prepariamo la richiesta per l'estrazione dei concetti

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    #inviamo la richiesta al server

    response.raise_for_status()
    #controlliamo eventuali errori http

    result = response.json()
    #trasformiamo la risposta in un dizionario python

    return result["choices"][0]["message"]["content"]
    #restituiamo il json generato da qwen


def acquire_concept(concept, example_images=None):
    #acquisiamo un nuovo concetto usando anche esempi visivi del dataset

    if example_images is None:
        example_images = []
    #se non ci sono immagini usiamo una lista vuota


    message_content = [
        {
            "type": "text",
            "text": f"""
We need to acquire a visual concept for a multimodal reasoning agent.

Concept:
{concept}

You are given example images from the visual environment in which
the agent will operate.

Some example images may contain the target concept and some may not.
Do NOT assume that every image contains the concept.

Use the images to understand how the concept actually appears
in this visual environment.

Create a representation with exactly these fields:

1. definition:
   the semantic meaning of the concept.

2. visual_cues:
   observable characteristics that help recognize the concept.

3. appearance_variants:
   ways the same concept may appear because of scale,
   rasterization, resolution, color, or other visual conditions.

4. common_confusions:
   visually similar patterns that could be confused with the concept.

IMPORTANT:
- Do not describe an idealized mathematical object only.
- Base visual observations on the provided images.
- Small objects may have pixelated or jagged boundaries.
- Do not invent properties that are not supported by the examples.
- Do not solve a classification task.

Return JSON only using exactly this structure:

{{
    "definition": "",
    "visual_cues": [],
    "appearance_variants": [],
    "common_confusions": []
}}
"""
        }
    ]
    #costruiamo il contenuto testuale della richiesta


    for image_path in example_images:
        #aggiungiamo ogni immagine alla richiesta multimodale

        image_url = image_to_data_url(image_path)
        #convertiamo l'immagine in una data URL

        message_content = message_content + [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
        #aggiungiamo l'immagine alla lista dei contenuti


    payload = {
        "model": MODEL_NAME,

        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ],

        "temperature": 0,

        "max_tokens": 300,

        "response_format": {
            "type": "json_object"
        }
    }
    #prepariamo la richiesta multimodale


    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    #inviamo la richiesta a llama-server


    response.raise_for_status()
    #solleviamo un errore se il server restituisce un errore HTTP


    result = response.json()
    #trasformiamo la risposta in un dizionario Python


    return result["choices"][0]["message"]["content"]
    #restituiamo la rappresentazione acquisita

def acquire_relation(relation):
    prompt = f"""
We need to acquire a new semantic relation for a visual reasoning agent.

Relation:
{relation}

Return JSON only using exactly this structure:

{{
    "definition": "",
    "visual_cues": [],
    "argument_structure": ""
}}

The definition should explain the visual meaning of the relation.
The visual_cues should describe how the relation can be recognized.
The argument_structure should describe what entities participate in
the relation.

Do not discuss any specific image.
Only describe the relation.
"""
    #costruiamo il prompt per acquisire una nuova relazione

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {
            "type": "json_object"
        }
    }
    #prepariamo la richiesta per qwen

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    #inviamo la richiesta al server

    response.raise_for_status()
    #controlliamo eventuali errori http

    result = response.json()
    #trasformiamo la risposta in un dizionario python

    return result["choices"][0]["message"]["content"]
    #restituiamo la rappresentazione della relazione


def _parse_json_response(text):
    #estraiamo il primo oggetto JSON completo anche se qwen aggiunge testo extra

    text = text.strip()
    #rimuoviamo spazi inutili

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    #proviamo prima il parsing diretto

    start = text.find("{")
    #cerchiamo l'inizio del json

    end = text.rfind("}")
    #cerchiamo la fine del json

    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    #proviamo a recuperare il json da eventuale testo circostante

    raise json.JSONDecodeError(
        "Qwen did not return valid JSON",
        text,
        0
    )
    #se non troviamo un json valido segnaliamo l'errore


def diagnose_error(
    image_path,
    question,
    predicted,
    ground_truth,
    evidence,
    required_concepts,
    current_memory=None
):
    #diagnostichiamo l'errore osservando anche la stessa immagine che ha causato l'errore

    if current_memory is None:
        current_memory = {
            "concepts": {},
            "relations": {}
        }
    #se non passiamo la memoria corrente usiamo una memoria vuota

    memory_text = json.dumps(
        current_memory,
        indent=2,
        ensure_ascii=False
    )
    #trasformiamo la memoria corrente in testo

    image_url = image_to_data_url(image_path)
    #prepariamo l'immagine dell'episodio che vogliamo diagnosticare

    prompt = f"""
Analyze this incorrect visual classification.

Task: {question}
Required concepts: {required_concepts}
Predicted: {predicted}
Correct: {ground_truth}
Previous evidence: {evidence}
Memory: {memory_text}

Inspect the image directly and identify whether the stored concept is insufficient.
Use should_refine=true only for a conceptual or visual-representation error.
Use should_refine=false for low_resolution, occlusion or ambiguous_image when the concept itself is not wrong.

Return ONLY valid JSON, with short strings and at most 3 items per list:
{{
  "error_type": "",
  "target_concept": "",
  "target_relation": "",
  "reason": "",
  "should_refine": false,
  "refined_definition": "",
  "refined_visual_cues": [],
  "refined_appearance_variants": [],
  "refined_common_confusions": []
}}
"""
    #costruiamo un prompt molto compatto per ridurre il rischio di output troncato

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 220,
        "response_format": {
            "type": "json_object"
        }
    }
    #prepariamo la richiesta multimodale della diagnosi

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=300
    )
    #inviamo la richiesta a llama-server

    response.raise_for_status()
    #controlliamo eventuali errori http

    result = response.json()
    #trasformiamo la risposta in un dizionario python

    raw_content = result["choices"][0]["message"]["content"]
    #recuperiamo il testo prodotto da qwen

    return raw_content
    #restituiamo direttamente l'output di qwen e lasciamo il parsing al chiamante


if __name__ == "__main__":

    question = "Does the image contain a square?"
    #domanda di test per l'estrazione dei concetti

    concepts = extract_concepts(question)
    #estraiamo i concetti dalla domanda

    print("Qwen concept extraction:")
    print(concepts)
