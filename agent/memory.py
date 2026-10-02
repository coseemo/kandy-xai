import copy
import json
from pathlib import Path


class ConceptMemory:
    #classe che si occupa di gestire la memoria dell'agente

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        #salviamo il percorso della memoria oppure None se vogliamo una memoria solo in ram

        if self.path is not None and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            #se esiste gia una memoria in formato json la carichiamo

        else:
            self.data = {
                "concepts": {},
                "relations": {},
                "history": []
            }
            #altrimenti creiamo una memoria vuota

        self.data.setdefault("concepts", {})
        #assicuriamo che esista la sezione dei concetti anche nelle vecchie memorie

        self.data.setdefault("relations", {})
        #assicuriamo che esista la sezione delle relazioni

        self.data.setdefault("history", [])
        #assicuriamo che esista lo storico degli episodi

    @classmethod
    def empty(cls):
        return cls(None)
        #creiamo una memoria temporanea che non viene scritta sul disco

    def get_concept(self, concept):
        return self.data["concepts"].get(concept)
        #cerchiamo un concetto nella memoria

    def get_relation(self, relation):
        return self.data["relations"].get(relation)
        #cerchiamo una relazione nella memoria

    def get(self, concept):
        return self.get_concept(concept)
        #manteniamo get come alias per compatibilita con il codice precedente

    def save_concept(
        self,
        concept,
        information,
        source="acquisition",
        task_id=None,
        sample_id=None
    ):
        info = copy.deepcopy(information)
        #creiamo una copia per non modificare il dizionario originale

        info.setdefault("error_history", [])
        #assicuriamo che ogni concetto abbia uno storico degli errori

        previous = self.data["concepts"].get(concept, {})
        #recuperiamo la versione precedente del concetto se esiste

        previous_meta = previous.get("_meta", {})
        #recuperiamo i metadati della versione precedente

        version = previous_meta.get("version", 0) + 1
        #aumentiamo il numero di versione

        info["_meta"] = {
            "version": version,
            "source": source,
            "task_id": task_id,
            "sample_id": sample_id
        }
        #salviamo anche la provenienza e la versione del concetto

        self.data["concepts"][concept] = info
        #salviamo il concetto nella memoria

        self.save()
        #scriviamo la memoria sul disco se la memoria e persistente

    def refine_concept(
        self,
        concept,
        information,
        reason,
        task_id=None,
        sample_id=None,
        error_type=None,
        evidence=None,
        predicted=None,
        ground_truth=None
    ):
        previous = self.data["concepts"].get(concept, {})
        #recuperiamo la rappresentazione precedente

        updated = copy.deepcopy(previous)
        #creiamo una copia della rappresentazione precedente

        for key in ["definition", "visual_cues", "appearance_variants", "common_confusions"]:
            if key in information:
                updated[key] = information[key]
        #aggiorniamo solo i campi concettuali che il raffinamento fornisce

        previous_meta = previous.get("_meta", {})
        #recuperiamo i metadati precedenti

        version = previous_meta.get("version", 0) + 1
        #aumentiamo la versione dopo il raffinamento

        error_history = copy.deepcopy(
            previous.get("error_history", [])
        )
        #recuperiamo lo storico degli errori gia osservati

        error_history.append({
            "error_type": error_type,
            "reason": reason,
            "task_id": task_id,
            "sample_id": sample_id,
            "evidence": evidence,
            "predicted": predicted,
            "ground_truth": ground_truth
        })
        #aggiungiamo l'errore che ha motivato il refinement

        updated["error_history"] = error_history
        #salviamo l'errore direttamente dentro la memoria del concetto

        updated["_meta"] = {
            "version": version,
            "source": "refinement",
            "task_id": task_id,
            "sample_id": sample_id,
            "reason": reason,
            "previous_version": previous_meta.get("version", 0)
        }
        #registriamo perche il concetto e stato modificato

        self.data["concepts"][concept] = updated
        #sostituiamo la vecchia rappresentazione con quella raffinata

        self.save()
        #rendiamo persistente il raffinamento

    def save_relation(
        self,
        relation,
        information,
        source="acquisition",
        task_id=None,
        sample_id=None
    ):
        info = copy.deepcopy(information)
        #creiamo una copia per non modificare il dizionario originale

        previous = self.data["relations"].get(relation, {})
        #recuperiamo la relazione precedente se esiste

        previous_meta = previous.get("_meta", {})
        #recuperiamo i metadati precedenti

        version = previous_meta.get("version", 0) + 1
        #aumentiamo il numero di versione

        info["_meta"] = {
            "version": version,
            "source": source,
            "task_id": task_id,
            "sample_id": sample_id
        }
        #salviamo anche la provenienza della relazione

        self.data["relations"][relation] = info
        #salviamo la relazione nella memoria

        self.save()
        #scriviamo la memoria sul disco

    def log_episode(self, episode):
        self.data["history"].append(copy.deepcopy(episode))
        #aggiungiamo l'episodio allo storico della memoria

        self.save()
        #rendiamo persistente lo storico

    def snapshot(self):
        return copy.deepcopy(self.data)
        #restituiamo una copia completa della memoria

    def save(self):
        if self.path is None:
            return
        #una memoria temporanea non viene scritta sul disco

        self.path.parent.mkdir(parents=True, exist_ok=True)
        #creiamo la cartella della memoria se non esiste

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                self.data,
                f,
                indent=4,
                ensure_ascii=False
            )
        #scriviamo la memoria sul disco in formato json
