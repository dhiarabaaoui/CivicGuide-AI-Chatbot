# Nettoyage de la phase expérimentale

## Objectif

Le projet contenait de nombreuses variantes créées pendant l'optimisation : 21 prompts, 20 notebooks, plus de 50 configurations, plusieurs modèles de reranking et des centaines de sorties intermédiaires. Elles étaient utiles au moment des essais, mais rendaient le projet difficile à comprendre et occupaient plusieurs gigaoctets. Le nettoyage du 13 septembre 2026 conserve un seul chemin de référence et résume les essais rejetés dans `GUIDE_MEMOIRE_PROJET.md`.

## Éléments conservés

- Trois prompts actifs : `mandatory_claim_planner_v5.txt`, `mandatory_claim_plan_reviewer_v3.txt` et `mandatory_claim_realizer_v1.txt`.
- Huit notebooks structurants : données, chunking, benchmark, recherche hybride RRF, contexte, analyse d'erreurs, préparation production et évaluation de bout en bout.
- Une seule version retenue des principaux artefacts volumineux : chunks, benchmark, BM25, recherche dense, recherche hybride, contexte et contrat de génération.
- Le candidat gelé `rag_candidate_structural_v1_62383ac6c4ff` et son snapshot complet.
- Les embeddings, car leur recalcul nécessiterait de nouveaux appels payants.
- Les données et rapports de validation externe, les décisions du propriétaire et les métriques finales, y compris les échecs.
- Le code nécessaire au pipeline actif, aux tests et à l'audit externe.

## Éléments retirés du projet actif

- Les anciennes versions de prompts et leurs canaris v2, v3, v4, v7 et v8.
- Les notebooks redondants dont la conclusion est déjà intégrée au guide.
- Les configurations et scripts propres aux branches rejetées.
- Le cross-encoder de reranking, son cache et ses résultats, car cette branche a été rejetée après une amélioration insuffisante et une régression du MRR.
- Les contrôleurs, planners alternatifs, réécritures de requêtes, quality gates et optimisations de préproduction non retenus.
- Les anciennes versions d'artefacts quand une version finale équivalente est conservée.

## Sens de « réduire les boucles »

Le nettoyage supprime les boucles de recherche déjà terminées, c'est-à-dire l'accumulation de variantes, de notebooks et de résultats. Il ne retire pas les contrôles de sûreté du pipeline actif. Les reprises réseau, la validation du contrat et les tentatives limitées de réparation restent nécessaires pour éviter qu'une erreur temporaire ou une sortie JSON invalide provoque une réponse incorrecte.

## Reproductibilité

Les 329 éléments retirés, représentant 2 523,90 Mo, ont été déplacés de façon récupérable dans un dossier d’archive frère nommé `RAG Assistant - archive experiments 2026-09-13`. Le fichier `archive_manifest.json` de ce dossier donne le chemin et la taille de chaque élément. Le projet actif est passé d'environ 3,63 Go à 1,10 Go de données, plus moins de 4 Mo de code, notebooks et rapports. Les principales conclusions restent aussi consignées dans `GUIDE_MEMOIRE_PROJET.md`. Aucun résultat négatif final n'a été transformé ou présenté comme un succès.

## Vérification après nettoyage

Après le nettoyage, les neuf configurations de recherche restantes étaient des JSON valides, les scripts conservés se compilaient et les 62 tests disponibles réussissaient. La phase suivante a ajouté séparément `configs/runtime.json` et ses tests. Le candidat gelé contient toujours ses onze artefacts. Ses empreintes correspondent au manifeste après normalisation des fins de ligne Windows CRLF vers LF ; le contenu logique du snapshot est donc intact.
