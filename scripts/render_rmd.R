#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop(
    "Usage: Rscript scripts/render_rmd.R <file.Rmd> [<file.Rmd> ...]"
  )
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

output_dir <- "html"
