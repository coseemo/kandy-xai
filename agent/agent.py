import json

from memory import ConceptMemory
from qwen_client import (
    extract_concepts,
    ask_qwen,
    diagnose_error,
    acquire_concept,
    acquire_relation
)
from kandy_loader import load_samples
#importiamo le funzioni che ci servono e la funzione che carica gli esempi di KANDY


def retrieve_memory(
    question,
    memory,
    acquire_unknown=True,
    extracted=None,
    task_id=None,
    sample_id=None,
    acquisition_images=None
):
    #estraiamo i concetti e recuperiamo o acquisiamo quelli necessari

    if extracted is None:
        raw = extract_concepts(question)
        extracted = json.loads(raw)
        #se non abbiamo gia l'estrazione chiediamo a qwen di farla

    if acquisition_images is None:
        acquisition_images = []
        #acquisition_images contiene gli esempi visivi usati per imparare il concetto

    concepts = extracted.get("concepts", [])
    #prendiamo i concetti estratti

    relations = extracted.get("relations", [])
    #prendiamo le relazioni estratte

    memories = {}
    #creiamo un dizionario con i concetti che passeremo a qwen

    relation_memories = {}
    #creiamo un dizionario con le relazioni che passeremo a qwen

    newly_acquired = []
    #teniamo traccia dei concetti e delle relazioni acquisiti in questo episodio

    for concept in concepts:

        information = memory.get_concept(concept)
        #cerchiamo il concetto nella memoria

        if information is None:

            if acquire_unknown:

                raw_concept = acquire_concept(
                    concept,
                    example_images=acquisition_images,
                )
                #acquisiamo il concetto usando esempi visivi del dataset

                concept_information = json.loads(raw_concept)
                #trasformiamo la rappresentazione del concetto in un dizionario python

                memory.save_concept(
                    concept,
                    concept_information,
                    source="acquisition",
                    task_id=task_id,
                    sample_id=sample_id
                )
                #salviamo il nuovo concetto nella memoria

                information = memory.get_concept(concept)
                #recuperiamo il concetto appena salvato

                newly_acquired.append(concept)
                #registriamo il nuovo concetto

                print(f"\nConcept acquired: {concept}")
                #mostriamo quale concetto e stato acquisito

            else:

                continue
                #se non vogliamo acquisire il concetto lo lasciamo fuori dalla memoria

        memories[concept] = information
        #aggiungiamo il concetto alla conoscenza passata a qwen

    for relation in relations:

        information = memory.get_relation(relation)
        #cerchiamo la relazione nella memoria

        if information is None:

            if acquire_unknown:

                raw_relation = acquire_relation(relation)
                #se la relazione non esiste la facciamo acquisire a qwen

                relation_information = json.loads(raw_relation)
                #trasformiamo la rappresentazione della relazione in un dizionario python

                memory.save_relation(
                    relation,
                    relation_information,
                    source="acquisition",
                    task_id=task_id,
                    sample_id=sample_id
                )
                #salviamo la nuova relazione nella memoria

                information = memory.get_relation(relation)
                #recuperiamo la relazione appena salvata

                newly_acquired.append(relation)
                #registriamo anche la nuova relazione

                print(f"\nRelation acquired: {relation}")
                #mostriamo quale relazione e stata acquisita

            else:

                continue
                #se non vogliamo acquisire la relazione la lasciamo fuori

        relation_memories[relation] = information
        #aggiungiamo la relazione alla conoscenza passata a qwen

    knowledge = {
        "concepts": memories,
        "relations": relation_memories
    }
    #costruiamo la conoscenza esterna che passera a qwen

    return concepts, relations, knowledge, newly_acquired


def process_sample(
    sample,
    memory,
    acquire_unknown=True,
    auto_refine=False,
    extracted=None
):
    #eseguiamo un episodio completo dell'agente

    image_path = sample["image_path"]
    #prendiamo l'immagine

    question = sample["question"]
    #prendiamo la domanda

    label = sample["label"]
    #prendiamo il ground truth

    concepts, relations, knowledge, newly_acquired = retrieve_memory(
        question,
        memory,
        acquire_unknown=acquire_unknown,
        extracted=extracted,
        task_id=sample.get("task_id"),
        sample_id=sample.get("image_path")
    )
    #recuperiamo o acquisiamo i concetti prima della classificazione

    answer = ask_qwen(
        image_path,
        question,
        knowledge
    )
    #mandiamo a qwen immagine, domanda e conoscenza recuperata

    result = json.loads(answer)
    #trasformiamo la risposta json in un dizionario python

    prediction = result["answer"]
    #prendiamo la classificazione

    used_concepts = [
        concept
        for concept in result.get("used_concepts", [])
        if concept in knowledge["concepts"]
    ]
    #teniamo solo i concetti dichiarati come usati e presenti nella memoria passata a qwen

    used_relations = [
        relation
        for relation in result.get("used_relations", [])
        if relation in knowledge["relations"]
    ]
    #teniamo solo le relazioni dichiarate come usate e presenti nella memoria

    evidence = result.get("evidence", "")
    #recuperiamo l'evidenza testuale

    diagnosis = None
    #nessun refinement e stato applicato all'inizio dell'episodio

    refinement_applied = False
    #teniamo esplicitamente traccia dell\'aggiornamento reale della memoria

    if prediction != label and auto_refine:
        diagnosis_raw = ""
        #inizializziamo la risposta grezza per poterla registrare anche in caso di errore

        try:
            diagnosis_raw = diagnose_error(
                image_path,
                question,
                prediction,
                label,
                evidence,
                concepts,
                knowledge
            )
            #se sbagliamo chiediamo a qwen una diagnosi dell'errore

            diagnosis = json.loads(diagnosis_raw)
            #trasformiamo la diagnosi in un dizionario python

        except json.JSONDecodeError as exc:
            diagnosis = {
                "error_type": "diagnosis_parse_error",
                "target_concept": "",
                "target_relation": "",
                "reason": "Qwen did not return valid JSON.",
                "should_refine": False,
                "raw_output": diagnosis_raw,
                "parse_error": str(exc)
            }
            #non blocchiamo l'esperimento se qwen produce un json non valido

        target_concept = diagnosis.get("target_concept", "")
        #prendiamo il concetto che la diagnosi ritiene da raffinare

        should_refine = diagnosis.get("should_refine", False)
        #controlliamo se la diagnosi propone davvero un raffinamento

        if (
            should_refine
            and target_concept in knowledge["concepts"]
        ):
            refined_information = {
                "definition": diagnosis.get(
                    "refined_definition",
                    knowledge["concepts"][target_concept].get("definition", "")
                ),
                "visual_cues": diagnosis.get(
                    "refined_visual_cues",
                    knowledge["concepts"][target_concept].get("visual_cues", [])
                ),
                "appearance_variants": diagnosis.get(
                    "refined_appearance_variants",
                    knowledge["concepts"][target_concept].get("appearance_variants", [])
                ),
                "common_confusions": diagnosis.get(
                    "refined_common_confusions",
                    knowledge["concepts"][target_concept].get("common_confusions", [])
                )
            }
            #costruiamo la nuova rappresentazione del concetto

            memory.refine_concept(
                target_concept,
                refined_information,
                diagnosis.get("reason", ""),
                task_id=sample.get("task_id"),
                sample_id=sample.get("image_path"),
                error_type=diagnosis.get("error_type", ""),
                evidence=evidence,
                predicted=prediction,
                ground_truth=label
            )
            #salviamo il raffinamento nella memoria

            refinement_applied = True
            #il refinement e stato realmente applicato alla memoria

    record = {
        "image": image_path,
        "task_id": sample.get("task_id"),
        "question": question,
        "label": label,
        "required_concepts": concepts,
        "required_relations": relations,
        "newly_acquired": newly_acquired,
        "provided_concepts": list(knowledge["concepts"].keys()),
        "provided_relations": list(knowledge["relations"].keys()),
        "used_concepts": used_concepts,
        "used_relations": used_relations,
        "prediction": prediction,
        "correct": prediction == label,
        "evidence": evidence,
        "diagnosis": diagnosis,
        "refinement_applied": refinement_applied
    }
    #costruiamo il log strutturato dell'episodio

    return record


if __name__ == "__main__":

    memory = ConceptMemory("memory/concepts.json")
    #carichiamo la memoria esterna

    samples = load_samples()
    #carichiamo gli esempi di KANDY

    sample = next(
        sample for sample in samples
        if sample["image_path"].endswith("0/000278.png")
    )
    #selezioniamo il campione che useremo per il test

    record = process_sample(
        sample,
        memory,
        acquire_unknown=True,
        auto_refine=True
    )
    #eseguiamo un episodio completo

    print("\nImage:")
    print(record["image"])

    print("\nQuestion:")
    print(record["question"])

    print("\nGround truth:")
    print(record["label"])

    print("\nRequired concepts:")
    print(record["required_concepts"])

    print("\nRequired relations:")
    print(record["required_relations"])

    print("\nNew concepts acquired:")
    print(record["newly_acquired"])

    print("\nProvided concepts:")
    print(record["provided_concepts"])

    print("\nQwen prediction:")
    print(record["prediction"])

    print("\nUsed concepts:")
    print(record["used_concepts"])

    print("\nUsed relations:")
    print(record["used_relations"])

    print("\nEvidence:")
    print(record["evidence"])

    if record["diagnosis"] is not None:
        print("\nError diagnosis:")
        print(json.dumps(
            record["diagnosis"],
            indent=4,
            ensure_ascii=False
        ))
        #mostriamo la diagnosi solo quando l'esempio e sbagliato
