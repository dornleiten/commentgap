# Load the saved, canonical objects used by the CG1 R Markdown analyses.
# This file performs reads and inexpensive in-memory reconstruction only.

commentgap_frozen_read_csv <- function(path, variable) {
  if (!file.exists(path)) {
    stop("Frozen input for `", variable, "` is missing: ", path,
         call. = FALSE)
  }
  data.table::fread(path)
}

commentgap_frozen_read_json <- function(path, variable) {
  if (!file.exists(path)) {
    stop("Frozen input for `", variable, "` is missing: ", path,
         call. = FALSE)
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

commentgap_frozen_common_inputs <- function(paths, document) {
  model_root <- commentgap_read_root(paths, "model_data")
  feature_root <- commentgap_read_root(paths, "features")
  split_path <- file.path(model_root, "master_article_split.parquet")
  manifest_path <- file.path(model_root, "feature_manifest.json")
  parameters_path <- file.path(model_root, "preprocessing_parameters.json")
  provenance_path <- file.path(feature_root, "provenance_manifest.json")
  required <- c(split_path, manifest_path, parameters_path, provenance_path)
  missing <- required[!file.exists(required)]
  if (length(missing)) {
    stop("Frozen inputs required by `", document,
         "` are missing: ", paste(missing, collapse = "; "), call. = FALSE)
  }
  article_split <- data.table::as.data.table(arrow::read_parquet(split_path))
  if ("story_id" %in% names(article_split)) {
    article_split[, story_id := as.character(story_id)]
  }
  list(
    model_data_root = model_root,
    source_feature_root = feature_root,
    split_path = split_path,
    manifest_path = manifest_path,
    parameters_path = parameters_path,
    provenance_path = provenance_path,
    article_split = article_split,
    feature_manifest = commentgap_frozen_read_json(manifest_path, "feature_manifest"),
    preprocessing_parameters = commentgap_frozen_read_json(
      parameters_path, "preprocessing_parameters"
    ),
    provenance = commentgap_frozen_read_json(provenance_path, "provenance")
  )
}

commentgap_frozen_results <- function(paths, document, scopes = c("root", "all")) {
  root_key <- if (document == "07_stacked_selection_models") {
    "frozen_cg1_regression"
  } else {
    "frozen_cg1_sensitivity"
  }
  artifact_root <- commentgap_read_root(paths, root_key)

  read_csv <- function(scope, file, variable) {
    commentgap_frozen_read_csv(
      file.path(artifact_root, scope, file), paste0(variable, " (", scope, ")")
    )
  }

  results <- lapply(scopes, function(scope) {
    if (document == "07_stacked_selection_models") {
      list(
        scope = scope,
        associations = read_csv(scope, "selector_associations.csv", "associations"),
        probabilities = read_csv(scope, "probability_contrasts.csv", "probabilities"),
        summary = read_csv(scope, "sample_summary.csv", "summary")
      )
    } else if (document == "07A1_collinearity_sensitivity") {
      version_root <- file.path(artifact_root, "collinearity_shared_preprocessing_v4")
      read_version_csv <- function(file, variable) {
        commentgap_frozen_read_csv(
          file.path(version_root, scope, file), paste0(variable, " (", scope, ")")
        )
      }
      list(
        scope = scope,
        covariance = read_version_csv("covariance_conditioning.csv", "covariance"),
        coefficient_pairs = read_version_csv("high_coefficient_correlations.csv", "coefficient_pairs"),
        condition_summary = read_version_csv("design_condition_summary.csv", "condition_summary"),
        exact_checks = read_version_csv("exact_reparameterisation_checks.csv", "exact_checks"),
        block_tests = read_version_csv("joint_block_wald_tests.csv", "block_tests"),
        cv_summary = read_version_csv("development_cv_metric_summary.csv", "cv_summary"),
        cv_fit_warnings = read_version_csv("development_cv_fit_warnings.csv", "cv_fit_warnings"),
        cv_differences = read_version_csv("development_cv_paired_differences.csv", "cv_differences")
      )
    } else if (document == "07A2_efron_exact_sensitivity") {
      version_root <- file.path(artifact_root, "efron_exact_shared_preprocessing_v4")
      read_version_csv <- function(file, variable) {
        commentgap_frozen_read_csv(
          file.path(version_root, scope, file), paste0(variable, " (", scope, ")")
        )
      }
      list(
        scope = scope,
        summary = read_version_csv("sensitivity_summary.csv", "summary"),
        comparison = read_version_csv("coefficient_comparison.csv", "comparison")
      )
    } else {
      stop("No frozen result loader for R Markdown document: ", document, call. = FALSE)
    }
  })
  names(results) <- scopes
  results
}
