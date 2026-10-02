import ast
import csv
from pathlib import Path


DATASET_ROOT = Path("/home/cosimo/kandy_agent/data")
#percorso principale del dataset KANDY


SETS_ROOT = DATASET_ROOT / "samples/sets"
#cartella che contiene train, val e test


TASK_QUESTIONS = {
    0: "Does the image contain a triangle?",
    1: "Does the image contain a square?",
    2: "Does the image contain a circle?",
    3: "Does the image contain a red object?",
    4: "Does the image contain a green object?",
    5: "Does the image contain a blue object?",
    6: "Does the image contain a cyan object?",
    7: "Does the image contain a magenta object?",
    8: "Does the image contain a yellow object?",
    9: "Does the image contain a red triangle on the right?",
    10: "Does the image contain a red triangle on the right and arbitrary objects?",
    11: "Does the image contain a red triangle on the right and at least one circle and arbitrary objects?",
    12: "Does the image contain a red triangle on the right and at least one blue object and arbitrary objects?",
    13: "Does the image contain a triangle and a square with the same color?",
    14: "Does the image contain a palindrome aba?",
    15: "Does the image contain a house?",
    16: "Does the image contain a car?",
    17: "Does the image contain a tower?",
    18: "Does the image contain a wagon?",
    19: "Does the image contain a traffic light?",
}
#domande associate ai venti task


SUPPORTED_GROUND_TRUTH_TASKS = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    13,
}
#task per cui abbiamo già una funzione di ground truth affidabile


def parse_symbol(raw_symbol):
    #trasformiamo il symbol del csv in una struttura Python

    if isinstance(raw_symbol, (dict, list)):
        return raw_symbol
    #se il symbol è già strutturato lo restituiamo direttamente

    if raw_symbol is None:
        return {}
    #gestiamo un eventuale symbol mancante

    raw_symbol = raw_symbol.strip()
    #rimuoviamo spazi inutili

    if not raw_symbol:
        return {}
    #gestiamo una stringa vuota

    return ast.literal_eval(raw_symbol)
    #convertiamo la stringa Python del csv in dict/list


def walk_symbol(value):
    #attraversiamo ricorsivamente tutta la struttura simbolica

    yield value
    #restituiamo anche il nodo corrente

    if isinstance(value, dict):

        for child in value.values():

            yield from walk_symbol(child)
            #visitiamo ricorsivamente tutti i valori del dizionario

    elif isinstance(value, list):

        for child in value:

            yield from walk_symbol(child)
            #visitiamo ricorsivamente tutti gli elementi della lista


def get_objects(symbol):
    #estraiamo gli oggetti visivi presenti nel symbol

    objects = []
    #lista degli oggetti

    for value in walk_symbol(symbol):

        if not isinstance(value, dict):
            continue
        #ignoriamo i valori che non sono dizionari

        if "shape" not in value:
            continue
        #un oggetto visivo deve avere almeno una shape

        objects.append({
            "shape": value.get("shape"),
            "color": value.get("color"),
            "size": value.get("size"),
        })
        #salviamo le proprietà dell'oggetto

    return objects
    #restituiamo tutti gli oggetti trovati


def has_shape(symbol, shape):
    #controlliamo se compare almeno un oggetto con la forma richiesta

    return any(
        obj["shape"] == shape
        for obj in get_objects(symbol)
    )
    #restituiamo True se troviamo la forma


def has_color(symbol, color):
    #controlliamo se compare almeno un oggetto con il colore richiesto

    return any(
        obj["color"] == color
        for obj in get_objects(symbol)
    )
    #restituiamo True se troviamo il colore


def triangle_and_square_same_color(symbol):
    #controlliamo il task triangle + square dello stesso colore

    objects = get_objects(symbol)
    #estraiamo tutti gli oggetti

    triangles = [
        obj
        for obj in objects
        if obj["shape"] == "triangle"
    ]
    #selezioniamo i triangoli

    squares = [
        obj
        for obj in objects
        if obj["shape"] == "square"
    ]
    #selezioniamo i quadrati

    for triangle in triangles:

        for square in squares:

            if triangle["color"] == square["color"]:
                return True
            #abbiamo trovato una coppia dello stesso colore

    return False
    #non esiste una coppia valida


def derive_ground_truth(task_id, symbol):
    #calcoliamo il ground truth usando solo il symbol


    if task_id == 0:

        return int(
            has_shape(symbol, "triangle")
        )
        #task 0: presenza di un triangolo


    if task_id == 1:

        return int(
            has_shape(symbol, "square")
        )
        #task 1: presenza di un quadrato


    if task_id == 2:

        return int(
            has_shape(symbol, "circle")
        )
        #task 2: presenza di un cerchio


    if task_id == 3:

        return int(
            has_color(symbol, "red")
        )
        #task 3: presenza di un oggetto rosso


    if task_id == 4:

        return int(
            has_color(symbol, "green")
        )
        #task 4: presenza di un oggetto verde


    if task_id == 5:

        return int(
            has_color(symbol, "blue")
        )
        #task 5: presenza di un oggetto blu


    if task_id == 6:

        return int(
            has_color(symbol, "cyan")
        )
        #task 6: presenza di un oggetto cyan


    if task_id == 7:

        return int(
            has_color(symbol, "magenta")
        )
        #task 7: presenza di un oggetto magenta


    if task_id == 8:

        return int(
            has_color(symbol, "yellow")
        )
        #task 8: presenza di un oggetto giallo


    if task_id == 13:

        return int(
            triangle_and_square_same_color(symbol)
        )
        #task 13: triangolo e quadrato dello stesso colore


    return None
    #per gli altri task il ground truth non è ancora implementato


def load_samples(split="train", task_id=None):
    #carichiamo uno split e, opzionalmente, un solo task

    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split deve essere 'train', 'val' oppure 'test'"
        )
    #controlliamo lo split


    split_root = SETS_ROOT / split
    #cartella dello split


    annotations_path = split_root / "annotations.csv"
    #file delle annotazioni


    if not annotations_path.exists():
        raise FileNotFoundError(
            f"File annotazioni non trovato: {annotations_path}"
        )
    #controlliamo che il file esista


    samples = []
    #lista dove salviamo gli esempi


    with open(
        annotations_path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)
        #leggiamo il CSV


        for row in reader:

            current_task_id = int(row["task_id"])
            #leggiamo il task dell'esempio


            if current_task_id not in TASK_QUESTIONS:
                continue
            #ignoriamo task sconosciuti


            if (
                task_id is not None
                and current_task_id != task_id
            ):
                continue
            #se richiesto un task specifico ignoriamo gli altri


            filename = row["filename"]
            #leggiamo il nome relativo dell'immagine


            image_path = split_root / filename
            #costruiamo il percorso completo


            symbol = parse_symbol(
                row.get("symbol", "")
            )
            #convertiamo il symbol in una struttura Python


            label = derive_ground_truth(
                current_task_id,
                symbol
            )
            #calcoliamo il ground truth quando il task è supportato


            supervised = (
                row.get("supervised", "")
                .strip()
                .lower()
                == "true"
            )
            #conserviamo il campo originale supervised


            sample = {
                "image_path": str(image_path),
                "task_id": current_task_id,
                "question": TASK_QUESTIONS[current_task_id],
                "label": label,
                "supervised": supervised,
                "symbol": symbol,
            }
            #costruiamo il campione


            samples.append(sample)
            #aggiungiamo il campione


    return samples
    #restituiamo tutti gli esempi richiesti


def get_task_samples(samples, task_id):
    #restituiamo tutti gli esempi appartenenti a un task

    return [
        sample
        for sample in samples
        if sample["task_id"] == task_id
    ]
    #filtriamo per task


def get_positive_samples(samples, task_id):
    #restituiamo gli esempi positivi di un task

    return [
        sample
        for sample in samples
        if (
            sample["task_id"] == task_id
            and sample["label"] == 1
        )
    ]
    #filtriamo i positivi


def get_negative_samples(samples, task_id):
    #restituiamo gli esempi negativi di un task

    return [
        sample
        for sample in samples
        if (
            sample["task_id"] == task_id
            and sample["label"] == 0
        )
    ]
    #filtriamo i negativi


def get_supported_samples(samples):
    #restituiamo solo i campioni con ground truth disponibile

    return [
        sample
        for sample in samples
        if sample["label"] is not None
    ]
    #eliminiamo i task non ancora implementati


if __name__ == "__main__":

    samples = load_samples(
        "test",
        task_id=2
    )
    #carichiamo solo il task 2 del test set


    print("\nTask 2 test samples:")
    print(len(samples))
    #mostriamo quanti esempi abbiamo


    print("\nLabels:")
    print(
        sorted({
            sample["label"]
            for sample in samples
        })
    )
    #controlliamo che siano presenti 0 e 1


    print("\nFirst 5 samples:")

    for sample in samples[:5]:

        print(sample)
        #mostriamo alcuni esempi per controllare il loader
