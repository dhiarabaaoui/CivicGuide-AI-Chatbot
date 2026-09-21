# Déploiement Vercel

## État

- Compte/scope : `dhia14`
- Projet : `civicguide-ai`
- Nom public : **CivicGuide AI**
- URL publique : <https://civicguide-ai-chatbot.vercel.app>
- Framework détecté : FastAPI
- Paquet source envoyé : environ 42,5 Mo
- Artefacts RAG : 1 273 chunks et vecteurs de dimension 1 536
- Authentification Vercel : désactivée pour ce projet afin de permettre la démonstration publique

Le frontend, le runtime et la génération réelle sont déployés. `OPENAI_API_KEY`
est enregistrée comme variable sensible dans les environnements Preview et
Production ; sa valeur n’est présente ni dans le dépôt ni dans cette
documentation. Le 20 septembre 2026, la page d’accueil et `/health` ont répondu
en HTTP 200, le runtime a signalé `status: ready` et un test réel de `/v1/chat`
a produit une réponse accompagnée de quatre citations.

## Préparer les artefacts

```powershell
python scripts/prepare_vercel_artifacts.py
python -m unittest discover -s tests -p "test_*.py" -q
```

Le script copie uniquement les quatre artefacts gelés requis dans
`runtime_artifacts/` et produit un manifeste avec leur taille et leur SHA-256.
Il refuse un paquet supérieur à 100 Mo.

## Vérifier le paquet sans le publier

```powershell
vercel deploy --dry --format=json
```

Le manifeste doit contenir environ 27 fichiers et ne doit jamais inclure
`.env`, `.env.local`, `data/`, `reports/`, `tests/` ou les notebooks.

## Variables et publication

`OPENAI_API_KEY` est déjà configurée comme secret pour Preview et Production.
Pour la remplacer, utiliser le tableau de bord Vercel ou `vercel env add`. Ne
jamais écrire sa valeur dans `vercel.json`, le dépôt ou la documentation. Après
toute modification d’une variable, créer un nouveau déploiement :

```powershell
vercel --prod
```

## Vérifications après publication

1. `GET /` doit afficher CivicGuide.
2. `GET /health` doit retourner `status: ready`,
   `generation_available: true`, 1 273 chunks et des vecteurs `[1273, 1536]`.
3. Le bouton de démonstration doit fonctionner sans appel OpenAI.
4. Une vraie question doit produire une réponse avec citations.
5. Une question posée dans le mauvais domaine doit être bloquée sans coût.

## Limites et protection des coûts

Vercel Hobby couvre l’hébergement dans ses quotas, mais les appels OpenAI restent
payants. Le runtime limite chaque requête générée à 0,03 USD. Pour une URL
publique, conserver ce plafond, surveiller l’usage OpenAI et configurer une règle
de limitation sur `/v1/chat` dans le pare-feu Vercel.
