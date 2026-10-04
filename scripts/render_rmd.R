#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop(
    "Usage: Rscript scripts/render_rmd.R <file.Rmd> [<file.Rmd> ...]"
  )
}

# Resolve explicit input paths against the caller before changing directories.
args <- normalizePath(args, mustWork = TRUE)

script_args <- commandArgs(trailingOnly = FALSE)
script_flag <- grep("^--file=", script_args, value = TRUE)
if (length(script_flag) != 1L) {
  stop("Could not determine the render wrapper path")
}
script_path <- normalizePath(
  sub("^--file=", "", script_flag[[1L]]),
  mustWork = TRUE
)
project_root <- normalizePath(
  file.path(dirname(script_path), ".."),
  mustWork = TRUE
)
setwd(project_root)

source(file.path(project_root, "scripts", "paths.R"))
paths <- commentgap_paths(project_root)
mode <- commentgap_mode()
if (!mode %in% c("frozen", "replay", "fresh", "resume")) {
  stop("COMMENTGAP_MODE must be replay, recompute, fresh, or resume")
}

if (!requireNamespace("rmarkdown", quietly = TRUE)) {
  stop("The rmarkdown package is required to render R Markdown documents")
}

missing <- args[!file.exists(args)]
if (length(missing)) {
  stop("Missing R Markdown file(s): ", paste(missing, collapse = ", "))
}

input_paths <- normalizePath(args, mustWork = TRUE)
extensions <- tolower(tools::file_ext(input_paths))
if (any(!extensions %in% c("rmd", "rmarkdown"))) {
  stop("All inputs must be R Markdown files (*.Rmd or *.Rmarkdown)")
}

output_names <- tools::file_path_sans_ext(basename(input_paths))
if (anyDuplicated(output_names)) {
  stop("Input R Markdown basenames must be unique")
}

sha256_file <- function(path) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("The digest package is required to verify frozen R Markdown artifacts")
  }
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

# Render destinations are staging; each scientific document below owns its
# input/option identity, allowing separate documents to share one run.
output_dir <- commentgap_stage(paths, "rendered/Rmd", kind = "staging")
read_stage_input <- function(key, env_var, run_area) {
  explicit <- Sys.getenv(env_var, "")
  candidate <- file.path(paths$root, paths$entries$outputs, commentgap_run_id(), run_area)
  if (!nzchar(explicit) && file.exists(candidate)) explicit <- candidate
  commentgap_read_root(paths, key, explicit = if (nzchar(explicit)) explicit else NULL)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (index in seq_along(input_paths)) {
  message("Rendering ", input_paths[[index]])
  if (mode %in% c("frozen", "replay")) {
    # The document's frozen branch loads canonical inputs and tabular analysis
    # products into its normal result objects; its usual display chunks render
    # those objects without running model fits or writing analysis artifacts.
    name <- output_names[[index]]
    intermediates_root <- file.path(output_dir, paste0(".", name, "-intermediates"))
    dir.create(intermediates_root, recursive = TRUE, showWarnings = FALSE)
    rmarkdown::render(
      input = input_paths[[index]],
      output_file = paste0(name, ".html"),
      output_dir = output_dir,
      intermediates_dir = intermediates_root,
      knit_root_dir = project_root,
      envir = new.env(parent = globalenv()),
      clean = TRUE,
      quiet = FALSE
    )
    message(if (mode == "replay") "Public replay rendered; open: " else "Private saved objects rendered; open: ",
            file.path(output_dir, paste0(name, ".html")))
    next
  }
  model_data <- read_stage_input("model_data", "COMMENTGAP_MODEL_DATA_ROOT", "shared/model_data")
  features <- read_stage_input("features", "COMMENTGAP_FEATURE_ROOT", "shared/features")
  Sys.setenv(COMMENTGAP_MODEL_DATA_ROOT = model_data, COMMENTGAP_FEATURE_ROOT = features)
  inputs <- c(input_paths[[index]], model_data, features)
  name <- output_names[[index]]
  if (identical(name, "07_stacked_selection_models")) {
    area <- "CG1/regression"
    variable <- "COMMENTGAP_REGRESSION_ROOT"
  } else if (identical(name, "07A1_collinearity_sensitivity")) {
    area <- "CG1/sensitivity/collinearity"
    variable <- "COMMENTGAP_COLLINEARITY_ROOT"
    regression <- read_stage_input("frozen_cg1_regression", "COMMENTGAP_REGRESSION_ROOT", "CG1/regression")
    Sys.setenv(COMMENTGAP_REGRESSION_ROOT = regression)
    inputs <- c(inputs, regression)
  } else if (identical(name, "07A2_efron_exact_sensitivity")) {
    area <- "CG1/sensitivity/efron_exact"
    variable <- "COMMENTGAP_REGRESSION_SENSITIVITY_ROOT"
  } else {
    stop("No scientific stage is registered for ", name)
  }
  explicit <- Sys.getenv(variable, "")
  stage_root <- commentgap_stage(
    paths, area, inputs = unique(inputs),
    contract = list(workflow = "rmarkdown", file = basename(input_paths[[index]]),
                    source_sha256 = sha256_file(input_paths[[index]])),
    explicit = if (nzchar(explicit)) explicit else NULL
  )
  do.call(Sys.setenv, setNames(list(stage_root), variable))
  old_stage_prepared <- Sys.getenv("COMMENTGAP_STAGE_PREPARED", unset = "")
  Sys.setenv(COMMENTGAP_STAGE_PREPARED = "1")
  intermediates_root <- file.path(output_dir, paste0(".", output_names[[index]], "-intermediates"))
  dir.create(intermediates_root, recursive = TRUE, showWarnings = FALSE)
  rmarkdown::render(
    input = input_paths[[index]],
    output_file = paste0(output_names[[index]], ".html"),
    output_dir = output_dir,
    intermediates_dir = intermediates_root,
    knit_root_dir = project_root,
    envir = new.env(parent = globalenv()),
    clean = TRUE,
    quiet = FALSE
  )
  if (nzchar(old_stage_prepared)) {
    Sys.setenv(COMMENTGAP_STAGE_PREPARED = old_stage_prepared)
  } else {
    Sys.unsetenv("COMMENTGAP_STAGE_PREPARED")
  }
}
