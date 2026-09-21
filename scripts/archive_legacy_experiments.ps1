param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectPrefix = $projectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$projectParent = Split-Path -Parent $projectRoot
$parentPrefix = $projectParent.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$archiveRoot = [IO.Path]::GetFullPath($ArchivePath)
$archivePrefix = $archiveRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar

if (-not $archiveRoot.StartsWith($parentPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The archive must be created under the project parent directory: $projectParent"
}
if ($archiveRoot.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The archive must be outside the active project directory."
}
if (Test-Path -LiteralPath $archiveRoot) {
    throw "The archive target already exists: $archiveRoot"
}

$keepPrompts = @(
    "mandatory_claim_planner_v5.txt",
    "mandatory_claim_plan_reviewer_v3.txt",
    "mandatory_claim_realizer_v1.txt"
)
$keepNotebooks = @(
    "01_data_analysis_validation.ipynb",
    "02_chunking_experiments.ipynb",
    "03_build_retrieval_benchmark.ipynb",
    "06_hybrid_rrf_retrieval_evaluation.ipynb",
    "09_context_builder_evaluation.ipynb",
    "12_error_analysis_adjudication.ipynb",
    "18_preproduction_ml_readiness.ipynb",
    "19_automated_end_to_end_evaluation.ipynb"
)
$keepConfigs = @(
    "bm25_experiments.json",
    "chunking_experiments.json",
    "context_builder_experiments.json",
    "dense_retrieval_experiments.json",
    "hybrid_retrieval_experiments.json",
    "mandatory_claim_pipeline.json",
    "production_release_decision.json",
    "retrieval_benchmark.json",
    "runtime_feature_flags.json"
)
$keepScripts = @(
    "analyze_completed_candidate_scoring.py",
    "analyze_existing_database_candidate_failures.py",
    "attach_existing_database_candidate_results.py",
    "audit_clean_validation_feasibility.py",
    "audit_fresh_validation_reference_quality.py",
    "build_ai_annotation_input.py",
    "build_planned_generation_contract.py",
    "claim_grounding.py",
    "collect_external_clean_validation_sources.py",
    "conversation_control.py",
    "conversation_state.py",
    "evaluate_mandatory_claim_pipeline.py",
    "evidence_anchor.py",
    "freeze_production_candidate.py",
    "import_external_reference_annotations.py",
    "plan_contract.py",
    "prepare_existing_database_question_pack.py",
    "prepare_external_validation_annotation_pack.py",
    "run_automated_end_to_end_evaluation.py",
    "run_external_locked_reference_candidate.py",
    "run_external_validation_annotation_app.ps1",
    "summarize_fresh_validation_pool.py",
    "validate_existing_database_question_pack.py",
    "validate_external_validation_annotation_pack.py",
    "archive_legacy_experiments.ps1"
)

$targets = @()
$targets += Get-ChildItem (Join-Path $projectRoot "prompts") -File | Where-Object Name -NotIn $keepPrompts
$targets += Get-ChildItem (Join-Path $projectRoot "notebooks") -File | Where-Object Name -NotIn $keepNotebooks
$targets += Get-ChildItem (Join-Path $projectRoot "configs") -File | Where-Object Name -NotIn $keepConfigs
$targets += Get-ChildItem (Join-Path $projectRoot "scripts") -File | Where-Object Name -NotIn $keepScripts
$targets += Get-ChildItem (Join-Path $projectRoot "scripts") -Directory -Filter "__pycache__"

$modelPath = Join-Path $projectRoot "models"
if (Test-Path -LiteralPath $modelPath) {
    $targets += Get-Item -LiteralPath $modelPath
}

$dataRoot = Join-Path $projectRoot "data/processed/multidoc2dial_v1"
$wholeDataDirectories = @(
    "answer_action_planner_experiments", "answer_evidence_planner_experiments",
    "answer_evidence_reranking_experiments", "claim_grounding_controller_experiments",
    "controlled_generation_experiments", "conversation_action_resolver_experiments",
    "conversation_state_cache", "conversation_state_experiments",
    "dialogue_transition_graph_experiments", "error_analysis_experiments",
    "evaluation_cache", "evidence_action_compatibility_experiments",
    "explicit_conversation_state_experiments", "fresh_human_canary_experiments",
    "generation_cache", "generation_experiments", "llm_action_adjudicator_experiments",
    "preproduction_context_experiments", "preproduction_optimization_experiments",
    "preproduction_rescue_reranking_experiments", "quality_evaluation_experiments",
    "query_rewriting_experiments", "reranker_cache", "reranking_experiments",
    "rewrite_cache", "runtime_quality_gate_experiments",
    "structural_context_rescue_experiments", "tri_action_planner_experiments"
)
foreach ($name in $wholeDataDirectories) {
    $path = Join-Path $dataRoot $name
    if (Test-Path -LiteralPath $path) { $targets += Get-Item -LiteralPath $path }
}

$dataChildKeep = @{
    "candidate_freezes" = @("rag_candidate_structural_v1_62383ac6c4ff")
    "chunking_experiments" = @("chunking_candidates_v0_b1ac9cc451")
    "retrieval_benchmarks" = @("multidoc2dial_retrieval_benchmark_v1_e8baf6e2ad")
    "bm25_experiments" = @("bm25_retrieval_baseline_v1_8def866155")
    "hybrid_retrieval_experiments" = @("hybrid_rrf_retrieval_v1_a91b993d86")
    "context_builder_experiments" = @("context_builder_v1_860d7c5b2f")
    "planned_generation_contract_experiments" = @("planned_generation_contract_v2_semantic_control_2f9f93633e")
    "mandatory_claim_pipeline_experiments" = @(
        "mandatory_claim_pipeline_v4_reviewed_coverage_full_39d45ad374",
        "mandatory_claim_pipeline_v5_semantic_direct_outcome_reserve_f9932692c1",
        "mandatory_claim_pipeline_v6_fragment_complete_reserve_c6530a994b"
    )
    "automated_end_to_end_evaluation_experiments" = @("automated_end_to_end_evaluation_v4_fresh_confirmation_30cd102ccd")
}
foreach ($parent in $dataChildKeep.Keys) {
    $path = Join-Path $dataRoot $parent
    if (Test-Path -LiteralPath $path) {
        $targets += Get-ChildItem -LiteralPath $path -Directory | Where-Object Name -NotIn $dataChildKeep[$parent]
    }
}

$reportRoot = Join-Path $projectRoot "reports/generated/multidoc2dial_v1"
$wholeReportDirectories = @(
    "action_controller_experiments", "answer_action_planner_experiments",
    "answer_evidence_planner_experiments", "answer_evidence_reranking_experiments",
    "claim_grounding_controller", "claim_grounding_controller_experiments",
    "controlled_generation_experiments", "conversation_action_resolver",
    "conversation_action_resolver_experiments", "conversation_control_experiments",
    "conversation_state_experiments", "dialogue_transition_graph_experiments",
    "error_analysis_experiments", "evidence_action_compatibility_experiments",
    "explicit_conversation_state", "explicit_conversation_state_experiments",
    "fresh_human_canary_experiments", "fresh_human_regression_replay_experiments",
    "generation_experiments", "known_human_regression_experiments",
    "llm_action_adjudicator_experiments", "plan_contract_guard_experiments",
    "planned_generation_canary_experiments", "preproduction_context_experiments",
    "preproduction_optimization_experiments", "preproduction_rescue_reranking_experiments",
    "quality_evaluation_experiments", "query_rewriting_experiments",
    "reranking_experiments", "runtime_quality_gate", "runtime_quality_gate_experiments",
    "runtime_quality_gate_final_reserve", "runtime_quality_gate_v2",
    "structural_context_rescue_experiments", "tri_action_planner_experiments"
)
foreach ($name in $wholeReportDirectories) {
    $path = Join-Path $reportRoot $name
    if (Test-Path -LiteralPath $path) { $targets += Get-Item -LiteralPath $path }
}

$reportChildKeep = $dataChildKeep.Clone()
$reportChildKeep.Remove("candidate_freezes")
foreach ($parent in $reportChildKeep.Keys) {
    $path = Join-Path $reportRoot $parent
    if (Test-Path -LiteralPath $path) {
        $targets += Get-ChildItem -LiteralPath $path -Directory | Where-Object Name -NotIn $reportChildKeep[$parent]
    }
}

$uniqueTargets = @($targets | Sort-Object FullName -Unique)
$manifestItems = foreach ($target in $uniqueTargets) {
    $source = [IO.Path]::GetFullPath($target.FullName)
    if ($source -eq $projectRoot -or -not $source.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe source path: $source"
    }
    $relative = $source.Substring($projectPrefix.Length)
    $destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot $relative))
    if (-not $destination.StartsWith($archivePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe destination path: $destination"
    }
    $bytes = if ($target.PSIsContainer) {
        (Get-ChildItem -LiteralPath $source -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    } else {
        $target.Length
    }
    [PSCustomObject]@{ relative_path = $relative; bytes = [long]$bytes; destination = $destination }
}

New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$manifest = [ordered]@{
    schema_version = "1.0.0"
    created_at = (Get-Date).ToString("o")
    source_project = $projectRoot
    archive_root = $archiveRoot
    target_count = $manifestItems.Count
    total_bytes = [long](($manifestItems | Measure-Object bytes -Sum).Sum)
    items = $manifestItems
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archiveRoot "archive_manifest.json") -Encoding utf8

foreach ($item in $manifestItems) {
    $source = Join-Path $projectRoot $item.relative_path
    $destinationParent = Split-Path -Parent $item.destination
    if (-not (Test-Path -LiteralPath $destinationParent)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
    Move-Item -LiteralPath $source -Destination $item.destination
}

[PSCustomObject]@{
    ArchivedTargets = $manifestItems.Count
    ArchivedMB = [math]::Round($manifest.total_bytes / 1MB, 2)
    ArchivePath = $archiveRoot
} | Format-List
