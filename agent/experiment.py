from __future__ import annotations

import json

from experiments._common import (
    CIRCLE_TEST_PROBES,
    ACQUISITION_IMAGES,
    binary_metrics,
    classify,
    fresh_memory,
    load_task_samples,
    print_prediction,
    sample_by_relpath,
    save_json,
    task_structure,
    acquire_one,
)
from agent import process_sample

TASK_ID = 2
#task 2 corrisponde a circle vs any


def main():
    #eseguiamo un esperimento piccolo ma completo su circle

    train = load_task_samples("train", TASK_ID)
    #carichiamo solo il training del task circle

    test = load_task_samples("test", TASK_ID)
    #carichiamo solo il test del task circle

    acquisition_sample = sample_by_relpath(
        "train",
        TASK_ID,
        ACQUISITION_IMAGES[TASK_ID],
    )
    #usiamo un solo esempio di acquisition e non 0000 che abbiamo visto essere difficile

    probes = [
        sample_by_relpath("test", TASK_ID, CIRCLE_TEST_PROBES[0]),
        sample_by_relpath("test", TASK_ID, CIRCLE_TEST_PROBES[1]),
    ]
    #usiamo solamente due immagini di test

    extracted = task_structure(acquisition_sample["question"])
    #estraiamo una sola volta la struttura del task

    print("\n====================================")
    print("1. BASELINE")
    print("====================================")
    #iniziamo dal baseline senza memoria

    baseline_memory = fresh_memory("task_2_baseline")
    #creiamo una memoria vuota

    baseline_records = []
    #salviamo le due predizioni baseline

    for sample in probes:
        result, _ = classify(sample, baseline_memory, extracted)
        #facciamo una sola inferenza per immagine

        print_prediction("BASELINE", sample, result)
        #mostriamo subito il risultato

        baseline_records.append({
            "image": sample["image_path"],
            "label": sample["label"],
            "prediction": result["answer"],
        })
        #salviamo la predizione

    print("\nBaseline metrics:")
    print(json.dumps(binary_metrics(baseline_records), indent=4))
    #mostriamo le metriche baseline

    print("\n====================================")
    print("2. ACQUISITION")
    print("====================================")
    #acquisiamo circle una sola volta

    memory = fresh_memory("task_2_refinement_experiment")
    #creiamo una memoria nuova per l'esperimento

    _, newly_acquired = acquire_one(
        TASK_ID,
        memory,
        acquisition_sample,
        extracted,
    )
    #acquisiamo il concetto usando una sola immagine

    print("Acquisition image:")
    print(acquisition_sample["image_path"])
    #mostriamo l'immagine usata

    print("Newly acquired:")
    print(newly_acquired)
    #mostriamo i concetti nuovi

    print("\n====================================")
    print("3. AFTER ACQUISITION")
    print("====================================")
    #misuriamo l'effetto della memoria senza fare refinement

    acquisition_records = []
    #salviamo le predizioni post-acquisition

    for sample in probes:
        result, _ = classify(sample, memory, extracted)
        #classifichiamo usando la memoria appena acquisita

        print_prediction("AFTER ACQUISITION", sample, result)
        #mostriamo il risultato

        acquisition_records.append({
            "image": sample["image_path"],
            "label": sample["label"],
            "prediction": result["answer"],
        })
        #salviamo la predizione

    print("\nAfter-acquisition metrics:")
    print(json.dumps(binary_metrics(acquisition_records), indent=4))
    #mostriamo le metriche dopo acquisition

    print("\n====================================")
    print("4. ERROR-DRIVEN REFINEMENT")
    print("====================================")
    #osserviamo una piccola sequenza di nuovi esempi di training dopo l'acquisition

    refinement_stream = [
        "2/0000.png",
        "2/0005.png",
        "2/0006.png",
        "2/0008.png",
        "2/0012.png",
    ]
    #usiamo pochi esempi per mantenere contenuti i tempi di esecuzione

    refinement_records = []
    #salviamo tutti gli episodi osservati durante il refinement stream

    refinement_record = None
    #indicheremo solo l'episodio in cui il refinement e stato realmente applicato

    for relpath in refinement_stream:
        hard_sample = sample_by_relpath("train", TASK_ID, relpath)
        #prendiamo un nuovo episodio di training senza riacquisire il concetto

        print(f"\nChecking refinement candidate: {hard_sample['image_path']}")
        #mostriamo il candidato corrente

        record = process_sample(
            hard_sample,
            memory,
            acquire_unknown=False,
            auto_refine=True,
            extracted=extracted,
        )
        #classifichiamo l'esempio e chiediamo una diagnosi solo in caso di errore

        refinement_records.append(record)
        #salviamo ogni episodio, anche quando non produce un refinement

        print(f"prediction={record['prediction']} label={record['label']}")
        #mostriamo il risultato della classificazione

        print(f"diagnosis={record['diagnosis']}")
        #mostriamo la diagnosi prodotta, se presente

        print(f"refinement_applied={record['refinement_applied']}")
        #mostriamo se la memoria e stata realmente modificata

        if record["refinement_applied"]:
            refinement_record = record
            #salviamo l'episodio che ha prodotto il refinement reale
            break
        #continuiamo con il prossimo episodio finche il refinement non avviene

    refinement_requested = any(
        record["diagnosis"] is not None
        and record["diagnosis"].get("should_refine", False)
        for record in refinement_records
    )
    #indichiamo se almeno una diagnosi ha richiesto un refinement

    refinement_applied = any(
        record["refinement_applied"]
        for record in refinement_records
    )
    #indichiamo se almeno un refinement e stato realmente applicato

    if refinement_records:
        print("\nMemory after refinement:")
        print(json.dumps(memory.snapshot(), indent=4, ensure_ascii=False))
        #mostriamo come e cambiata la memoria dopo lo stream

    print("\n====================================")
    print("5. TEST AFTER REFINEMENT")
    print("====================================")
    #ripetiamo lo stesso mini test dopo l'eventuale refinement

    final_records = []
    #salviamo le predizioni finali

    for sample in probes:
        result, _ = classify(sample, memory, extracted)
        #classifichiamo con la memoria aggiornata

        print_prediction("AFTER REFINEMENT", sample, result)
        #mostriamo il risultato finale

        final_records.append({
            "image": sample["image_path"],
            "label": sample["label"],
            "prediction": result["answer"],
        })
        #salviamo la predizione

    summary = {
        "task_id": TASK_ID,
        "acquisition_image": acquisition_sample["image_path"],
        "refinement_candidate": (
            refinement_record["image"] if refinement_record else None
        ),
        "refinement_requested": refinement_requested,
        "refinement_applied": refinement_applied,
        "refinement_stream": refinement_records,
        "baseline": binary_metrics(baseline_records),
        "after_acquisition": binary_metrics(acquisition_records),
        "after_refinement": binary_metrics(final_records),
        "acquisition_changed_accuracy": (
            binary_metrics(acquisition_records)["accuracy"]
            - binary_metrics(baseline_records)["accuracy"]
        ),
        "refinement_changed_accuracy": (
            binary_metrics(final_records)["accuracy"]
            - binary_metrics(acquisition_records)["accuracy"]
        ),
        "memory": memory.snapshot(),
        "refinement_record": refinement_record,
        "test_records": final_records,
        "train_samples_loaded": len(train),
        "test_samples_loaded": len(test),
    }
    #costruiamo il risultato compatto dell'esperimento

    path = save_json("task_2_fast_acquisition_refinement.json", summary)
    #salviamo il risultato

    print("\n====================================")
    print("FINAL RESULTS")
    print("====================================")
    #stampiamo un riepilogo finale

    print(json.dumps({
        "baseline": summary["baseline"],
        "after_acquisition": summary["after_acquisition"],
        "after_refinement": summary["after_refinement"],
        "refinement_requested": summary["refinement_requested"],
        "refinement_applied": summary["refinement_applied"],
        "results_file": str(path),
    }, indent=4))
    #mostriamo anche dove e stato salvato il risultato completo
    #mostriamo solo le informazioni principali


if __name__ == "__main__":
    main()
    #avviamo l'esperimento
