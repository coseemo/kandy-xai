from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#cartella principale del progetto

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
#permettiamo agli script di importare agent.py, memory.py, qwen_client.py e kandy_loader.py


RESULTS_DIR = PROJECT_ROOT / "results"
#cartella dove salviamo i risultati


def save_json(filename: str, data: dict[str, Any]) -> Path:
    #salviamo un risultato in formato JSON

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    #creiamo la cartella dei risultati se non esiste

    path = RESULTS_DIR / filename
    #costruiamo il percorso del file

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        #scriviamo il risultato

    return path
    #restituiamo il percorso del file


def choose_balanced(samples, n_positive: int, n_negative: int, seed: int = 12345):
    #selezioniamo un piccolo insieme bilanciato di test

    positive = [sample for sample in samples if sample["label"] == 1]
    #prendiamo i positivi

    negative = [sample for sample in samples if sample["label"] == 0]
    #prendiamo i negativi

    rng = random.Random(seed)
    #creiamo un generatore casuale riproducibile

    rng.shuffle(positive)
    #mescoliamo i positivi

    rng.shuffle(negative)
    #mescoliamo i negativi

    selected = positive[:n_positive] + negative[:n_negative]
    #selezioniamo il numero richiesto di campioni

    rng.shuffle(selected)
    #mescoliamo l'ordine finale

    return selected
    #restituiamo il mini test set


def metrics(results, prediction_key: str) -> dict[str, Any]:
    #calcoliamo le metriche binarie

    if not results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    #gestiamo il caso di lista vuota

    tp = sum(
        r[prediction_key] == 1 and r["label"] == 1
        for r in results
    )
    #true positive

    tn = sum(
        r[prediction_key] == 0 and r["label"] == 0
        for r in results
    )
    #true negative

    fp = sum(
        r[prediction_key] == 1 and r["label"] == 0
        for r in results
    )
    #false positive

    fn = sum(
        r[prediction_key] == 0 and r["label"] == 1
        for r in results
    )
    #false negative

    accuracy = (tp + tn) / len(results)
    #calcoliamo accuracy

    precision = tp / (tp + fp) if tp + fp else 0.0
    #calcoliamo precision

    recall = tp / (tp + fn) if tp + fn else 0.0
    #calcoliamo recall

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    #restituiamo le metriche


def acquire_task(task_id: int, memory, split: str = "train") -> dict[str, Any]:
    #acquisiamo tutti i concetti e le relazioni richiesti dalla domanda del task

    from kandy_loader import load_samples
    from qwen_client import extract_concepts
    from agent import retrieve_memory

    samples = load_samples(split, task_id=task_id)
    #carichiamo solo il task richiesto

    if not samples:
        raise ValueError(f"Nessun campione trovato per task {task_id} nello split {split}.")
    #controlliamo che il task esista

    question = samples[0]["question"]
    #prendiamo la domanda associata al task

    extracted = json.loads(extract_concepts(question))
    #estraiamo una sola volta la struttura simbolica richiesta

    acquisition_sample = next(
        sample
        for sample in samples
        if sample["image_path"].endswith(ACQUISITION_IMAGE)
    )
    #selezioniamo esplicitamente la stessa immagine di acquisition del test precedente

    _, _, knowledge, newly_acquired = retrieve_memory(
        question,
        memory,
        acquire_unknown=True,
        extracted=extracted,
        task_id=task_id,
        sample_id=acquisition_sample["image_path"],
        acquisition_images=[acquisition_sample["image_path"]],
    )
    #acquisiamo i concetti e le relazioni mancanti

    return {
        "question": question,
        "extracted": extracted,
        "knowledge": knowledge,
        "newly_acquired": newly_acquired,
        "acquisition_sample": acquisition_sample["image_path"],
    }
    #restituiamo tutte le informazioni dell'acquisizione


def run_condition(
    sample,
    memory,
    extracted,
    acquire_unknown=False,
    knowledge_override=None,
):
    #eseguiamo una classificazione con una specifica memoria

    from agent import retrieve_memory
    from qwen_client import ask_qwen

    if knowledge_override is None:
        _, _, knowledge, _ = retrieve_memory(
            sample["question"],
            memory,
            acquire_unknown=acquire_unknown,
            extracted=extracted,
            task_id=sample["task_id"],
            sample_id=sample["image_path"],
        )
        #recuperiamo normalmente la conoscenza filtrata dal task
    else:
        knowledge = knowledge_override
        #usiamo direttamente la conoscenza passata dalla condizione sperimentale

    raw = ask_qwen(
        sample["image_path"],
        sample["question"],
        knowledge,
    )
    #classifichiamo l'immagine usando la conoscenza fornita

    result = json.loads(raw)
    #convertiamo la risposta in un dizionario

    return result, knowledge, raw
    #restituiamo risposta, conoscenza e output grezzo di qwen


def available_task_samples(split: str, task_id: int):
    #carichiamo i campioni di un determinato task

    from kandy_loader import load_samples

    samples = load_samples(split, task_id=task_id)
    #carichiamo solo il task richiesto

    if not samples:
        raise ValueError(f"Nessun campione trovato per task {task_id} nello split {split}.")
    #controlliamo che esistano campioni

    return samples
    #restituiamo i campioni

from memory import ConceptMemory


TASK_ID = 2
#task di riferimento: circle

ACQUISITION_IMAGE = "2/0010.png"
#usiamo la stessa immagine positiva gia usata nel test precedente


N_POSITIVE = 5
#numero di positivi

N_NEGATIVE = 5
#numero di negativi

SEED = 12345
#seed riproducibile


def main():
    #eseguiamo l'ablation della memoria rilevante

    memory = ConceptMemory.empty()
    #partiamo da una memoria vuota

    acquisition = acquire_task(
        TASK_ID,
        memory,
        split="train",
    )
    #acquisiamo il concetto rilevante

    extracted = acquisition["extracted"]
    #conserviamo la struttura del task

    from kandy_loader import load_samples

    test = load_samples("test", task_id=TASK_ID)
    #carichiamo il test del task

    selected = choose_balanced(
        test,
        N_POSITIVE,
        N_NEGATIVE,
        seed=SEED,
    )
    #selezioniamo un piccolo test

    rows = []
    #risultati dei tre setting

    for sample in selected:

        relevant_result, relevant_knowledge, relevant_raw = run_condition(
            sample,
            memory,
            extracted,
            acquire_unknown=False,
        )
        #condizione con memoria rilevante

        empty_result, _, empty_raw = run_condition(
            sample,
            ConceptMemory.empty(),
            extracted,
            acquire_unknown=False,
        )
        #condizione senza memoria

        square_memory = ConceptMemory.empty()
        #creiamo una memoria con un concetto irrilevante ma appartenente allo stesso dominio visivo

        square_memory.save_concept(
            "square",
            {
                "definition": "A four-sided geometric shape.",
                "visual_cues": ["four straight sides", "four corners"],
                "appearance_variants": ["different colors and sizes"],
                "common_confusions": ["rectangle", "diamond"],
            },
            source="ablation_control",
        )
        #aggiungiamo esplicitamente square come controllo irrilevante

        square_knowledge = {
            "concepts": {
                "square": square_memory.get_concept("square")
            },
            "relations": {}
        }
        #costruiamo la memoria square che verra realmente fornita a qwen

        square_result, square_knowledge, square_raw = run_condition(
            sample,
            square_memory,
            extracted,
            acquire_unknown=False,
            knowledge_override=square_knowledge,
        )
        #condizione con memoria irrilevante ma visivamente pertinente al dominio

        cow_memory = ConceptMemory.empty()
        #creiamo una memoria con un concetto completamente fuori dal dominio del task

        cow_memory.save_concept(
            "cow",
            {
                "definition": "A domesticated bovine mammal.",
                "visual_cues": ["four legs", "large body", "head with ears and muzzle"],
                "appearance_variants": ["different coat colors and body sizes"],
                "common_confusions": ["horse", "bull"],
            },
            source="ablation_control",
        )
        #aggiungiamo esplicitamente cow come controllo fuori dominio

        cow_knowledge = {
            "concepts": {
                "cow": cow_memory.get_concept("cow")
            },
            "relations": {}
        }
        #costruiamo la memoria cow che verra realmente fornita a qwen

        cow_result, cow_knowledge, cow_raw = run_condition(
            sample,
            cow_memory,
            extracted,
            acquire_unknown=False,
            knowledge_override=cow_knowledge,
        )
        #condizione con memoria irrilevante e fuori dominio

        rows.append({
            "image": sample["image_path"],
            "label": sample["label"],
            "prediction_no_memory": empty_result["answer"],
            "prediction_relevant_memory": relevant_result["answer"],
            "prediction_square_memory": square_result["answer"],
            "prediction_cow_memory": cow_result["answer"],
            "evidence_no_memory": empty_result.get("evidence", ""),
            "evidence_relevant_memory": relevant_result.get("evidence", ""),
            "evidence_square_memory": square_result.get("evidence", ""),
            "evidence_cow_memory": cow_result.get("evidence", ""),
            "raw_no_memory": empty_raw,
            "raw_relevant_memory": relevant_raw,
            "raw_square_memory": square_raw,
            "raw_cow_memory": cow_raw,
            "no_memory_used_concepts": empty_result.get("used_concepts", []),
            "relevant_used_concepts": relevant_result.get("used_concepts", []),
            "square_used_concepts": square_result.get("used_concepts", []),
            "cow_used_concepts": cow_result.get("used_concepts", []),
            "square_provided_concepts": list(square_knowledge["concepts"].keys()),
            "cow_provided_concepts": list(cow_knowledge["concepts"].keys()),
        })
        #salviamo tutte le condizioni per il campione

    summary = {
        "task_id": TASK_ID,
        "newly_acquired": acquisition["newly_acquired"],
        "conditions": {
            "no_memory": "no external concept memory provided",
            "relevant_memory": "circle concept acquired from a real KANDY training image",
            "square_memory": "irrelevant in-domain concept injected as an ablation control",
            "cow_memory": "irrelevant out-of-domain concept injected as an ablation control",
        },
        "metrics_no_memory": metrics(rows, "prediction_no_memory"),
        "metrics_relevant_memory": metrics(
            rows,
            "prediction_relevant_memory",
        ),
        "metrics_square_memory": metrics(
            rows,
            "prediction_square_memory",
        ),
        "metrics_cow_memory": metrics(
            rows,
            "prediction_cow_memory",
        ),
        "rows": rows,
    }
    #costruiamo il riepilogo

    path = save_json(
        "03_memory_ablation.json",
        summary,
    )
    #salviamo i risultati

    print(json.dumps(summary, indent=4, ensure_ascii=False))
    #mostriamo il risultato

    print(f"\nSaved to: {path}")
    #mostriamo il file


if __name__ == "__main__":
    main()
