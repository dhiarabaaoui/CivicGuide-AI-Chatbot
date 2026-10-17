# Nettoyage de la phase expérimentale

## Objectif

Le projet contenait de nombreuses variantes créées pendant l'optimisation : 21 prompts, 20 notebooks, plus de 50 configurations, plusieurs modèles de reranking et des centaines de sorties intermédiaires. Elles étaient utiles au moment des essais, mais rendaient le projet difficile à comprendre et occupaient plusieurs gigaoctets. Le nettoyage du 13 septembre 2026 conserve un seul chemin de référence et résume les essais rejetés dans `GUIDE_MEMOIRE_PROJET.md`.

## Éléments conservés

- Trois prompts actifs : `mandatory_claim_planner_v5.txt`, `mandatory_claim_plan_reviewer_v3.txt` et `mandatory_claim_realizer_v1.txt`.
- Dix-sept notebooks documentaires et structurants après l'audit de conservation du 17 octobre 2026 : données, chunking, benchmark, baselines BM25 et dense, recherche hybride RRF, réécriture conditionnelle, reranking, contexte, génération, qualité, analyse d'erreurs, canary v2, préparation production et évaluation de bout en bout.
- Une seule version retenue des principaux artefacts volumineux : chunks, benchmark, BM25, recherche dense, recherche hybride, contexte et contrat de génération.
- Le candidat gelé `rag_candidate_structural_v1_62383ac6c4ff` et son snapshot complet.
- Les embeddings, car leur recalcul nécessiterait de nouveaux appels payants.
- Les données et rapports de validation externe, les décisions du propriétaire et les métriques finales, y compris les échecs.
- Le code nécessaire au pipeline actif, aux tests et à l'audit externe.

## Éléments retirés du projet actif

- Les anciennes versions de prompts et leurs boucles ciblées non retenues ; le canary v2 reste documenté parce qu'il introduit les affirmations atomiques reprises par l'architecture finale.
- Les notebooks 15 et 16, ciblés uniquement sur des échecs déjà connus, et le resolver 17 qui a échoué à ses gates d'intégration.
- Les configurations et scripts propres aux branches rejetées.
- Les poids, caches et résultats volumineux des cross-encoders, car cette branche a été rejetée après une amélioration insuffisante au niveau des preuves, une régression du MRR et un coût de latence élevé. Les notebooks de décision restent visibles.
- Les contrôleurs, planners alternatifs, sorties répétées de réécriture, quality gates et optimisations de préproduction non retenus.
- Les anciennes versions d'artefacts quand une version finale équivalente est conservée.

## Sens de « réduire les boucles »

Le nettoyage supprime les boucles de recherche déjà terminées, c'est-à-dire l'accumulation de variantes, de notebooks et de résultats. Il ne retire pas les contrôles de sûreté du pipeline actif. Les reprises réseau, la validation du contrat et les tentatives limitées de réparation restent nécessaires pour éviter qu'une erreur temporaire ou une sortie JSON invalide provoque une réponse incorrecte.

## Reproductibilité

Les 329 éléments retirés, représentant 2 523,90 Mo, ont été déplacés de façon récupérable dans un dossier d’archive frère nommé `RAG Assistant - archive experiments 2026-09-13`. Le fichier `archive_manifest.json` de ce dossier donne le chemin et la taille de chaque élément. Le projet actif est passé d'environ 3,63 Go à 1,10 Go de données, plus moins de 4 Mo de code, notebooks et rapports. Les principales conclusions restent aussi consignées dans `GUIDE_MEMOIRE_PROJET.md`. Aucun résultat négatif final n'a été transformé ou présenté comme un succès.

Un audit réalisé le 17 octobre 2026 a constaté que la règle initiale de réduction à huit notebooks avait aussi retiré des étapes qui avaient réellement amélioré le projet ou qui étaient nécessaires pour comprendre les notebooks suivants. Neuf notebooks et leurs sept configurations légères ont donc été restaurés dans le dépôt actif. Les caches, modèles et rapports volumineux restent archivés. La politique détaillée et la justification notebook par notebook se trouvent dans `docs/NOTEBOOK_RETENTION.md`.

## Vérification après nettoyage

Après le nettoyage, les neuf configurations de recherche restantes étaient des JSON valides, les scripts conservés se compilaient et les 62 tests disponibles réussissaient. La phase suivante a ajouté séparément `configs/runtime.json` et ses tests. Le candidat gelé contient toujours ses onze artefacts. Ses empreintes correspondent au manifeste après normalisation des fins de ligne Windows CRLF vers LF ; le contenu logique du snapshot est donc intact.
