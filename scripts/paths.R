# Repository-rooted paths for R Markdown entry points.
#
# Keep this dependency-light counterpart of commentgap_analysis.paths so R
# renders follow the same frozen/fresh/resume contract.

commentgap_repo_root <- function(script_path = NULL) {
  start <- if (is.null(script_path)) getwd() else dirname(normalizePath(script_path))
  current <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(current, "config", "paths.json"))) return(current)
    parent <- dirname(current)
    if (identical(parent, current)) stop("Could not find config/paths.json")
    current <- parent
  }
}

commentgap_paths <- function(repo_root = commentgap_repo_root()) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required for config/paths.json")
  config <- jsonlite::fromJSON(file.path(repo_root, "config", "paths.json"), simplifyVector = FALSE)
  if (!identical(config$schema_version, 1L)) stop("Unsupported config/paths.json schema")
  list(root = normalizePath(repo_root, mustWork = TRUE), entries = config$paths)
}

commentgap_python <- function(paths) {
  configured <- Sys.getenv("COMMENTGAP_PYTHON", "/tmp/commentgap-analysis-check/bin/python")
  if (!file.exists(configured)) configured <- Sys.which("python3")
  if (!nzchar(configured) || !file.exists(configured)) stop("Python is required for the shared path contract")
  configured
}

commentgap_contract <- function(paths, request) {
  request$repo_root <- paths$root
  bridge <- file.path(paths$root, "scripts", "path_contract.py")
  request_path <- tempfile("commentgap-path-request-", fileext = ".json")
  output_path <- tempfile("commentgap-path-output-", fileext = ".json")
  on.exit(unlink(c(request_path, output_path), force = TRUE), add = TRUE)
  writeLines(jsonlite::toJSON(request, auto_unbox = TRUE, null = "null", force = TRUE), request_path, useBytes = TRUE)
  status <- system2(commentgap_python(paths), c(bridge, "--request", request_path), stdout = output_path, stderr = output_path)
  output <- if (file.exists(output_path)) paste(readLines(output_path, warn = FALSE), collapse = "\n") else ""
  if (!identical(status, 0L)) stop("Shared Python path contract failed: ", output)
  parsed <- tryCatch(jsonlite::fromJSON(output, simplifyVector = FALSE), error = function(error) stop("Invalid path-contract response: ", output))
  parsed
}

commentgap_read_root <- function(paths, key, explicit = NULL, env_var = NULL) {
  result <- commentgap_contract(paths, list(
    command = "read_root", mode = commentgap_mode(), run_id = commentgap_run_id(),
    key = key, explicit = explicit, env_var = env_var, cwd = getwd()
  ))
  result$path
}

commentgap_mode <- function() {
  selected <- tolower(Sys.getenv("COMMENTGAP_MODE", "replay"))
  if (identical(selected, "recompute")) return("frozen")
  if (identical(selected, "frozen")) return("replay")
  selected
}

commentgap_run_id <- function() {
  value <- Sys.getenv("COMMENTGAP_RUN_ID", "")
  if (!nzchar(value)) NULL else value
}

commentgap_prepare_run <- function(paths, inputs = character(), contract = list()) {
  result <- commentgap_contract(paths, list(
    command = "prepare", mode = commentgap_mode(), run_id = commentgap_run_id(),
    inputs = as.list(as.character(inputs)), contract = contract, cwd = getwd()
  ))
  result$path
}

commentgap_stage <- function(paths, area, inputs = character(), contract = list(),
                             kind = "output", explicit = NULL) {
  result <- commentgap_contract(paths, list(
    command = "stage", mode = commentgap_mode(), run_id = commentgap_run_id(),
    area = area, kind = kind, explicit = explicit, inputs = as.list(as.character(inputs)),
    contract = contract, cwd = getwd()
  ))
  result$path
}

commentgap_stage_path <- function(paths, area, explicit = NULL) {
  result <- commentgap_contract(paths, list(
    command = "stage_path", mode = commentgap_mode(), run_id = commentgap_run_id(),
    area = area, explicit = explicit, cwd = getwd()
  ))
  result$path
}

commentgap_guard <- function(paths, destination) {
  result <- commentgap_contract(paths, list(
    command = "guard", mode = commentgap_mode(), run_id = commentgap_run_id(),
    destination = as.character(destination), cwd = getwd()
  ))
  result$path
}

commentgap_stage_child <- function(paths, stage_root, destination) {
  result <- commentgap_contract(paths, list(
    command = "stage_child", mode = commentgap_mode(), run_id = commentgap_run_id(),
    stage_root = as.character(stage_root), destination = as.character(destination), cwd = getwd()
  ))
  result$path
}

commentgap_staging_root <- function(paths, area) {
  kind <- if (commentgap_mode() %in% c("frozen", "replay")) "staging" else "output"
  commentgap_stage(paths, area, kind = kind)
}
