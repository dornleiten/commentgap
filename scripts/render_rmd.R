#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop(
    "Usage: Rscript scripts/render_rmd.R <file.Rmd> [<file.Rmd> ...]"
  )
}

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

output_dir <- "html"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (index in seq_along(input_paths)) {
  message("Rendering ", input_paths[[index]])
  rmarkdown::render(
    input = input_paths[[index]],
    output_file = paste0(output_names[[index]], ".html"),
    output_dir = output_dir,
    knit_root_dir = project_root,
    envir = new.env(parent = globalenv()),
    clean = TRUE,
    quiet = FALSE
  )
}
