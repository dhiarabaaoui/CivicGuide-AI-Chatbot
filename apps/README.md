# Applications locales

## API du RAG

Depuis la racine du projet, lancez :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rag_api.ps1
```

L'API est accessible sur <http://127.0.0.1:8000> et sa documentation sur <http://127.0.0.1:8000/docs>. Le fichier `apps/rag_api.py` expose la santé du service, la recherche seule et le chat contrôlé.

## Anciennes interfaces d'annotation

Les applications Streamlit présentes dans ce dossier servent uniquement à consulter ou poursuivre les audits humains historiques. Elles ne font pas partie du runtime utilisateur final.
