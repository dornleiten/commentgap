# Provenance records

This folder records the sources and validation of the retained run. Public
Replay notebooks read `artifacts/canonical/`, not this folder. The public files
here are:

- `paper-assets.csv`: submitted paper assets, their source stages, and hashes;
  used by publication validation.
- `environments/renv-original.lock`: the original R package snapshot, distinct
  from the current root `renv.lock`.
- `canonical-diagnostic-sample-20261003.json`: records the replacement
  diagnostic sample used for notebook 04.

Other records stay local and are excluded from Git. The large
`inventory.csv` and `artifact-files.csv.gz` list private file paths, sizes, and
hashes; they are not notebook inputs or a data backup. The private archive
manifest, bundle manifest, relocation registry, and migration records remain
at this level because local tools use those paths. `splits/` holds a private
article assignment file. `archive/` holds completed-run reports, superseded
listings, historical environment snapshots, and the superseded R startup
bootstrap. Its contents are
retained locally for audit but are not part of the public checkout.
Superseded standalone ranking launchers and the duplicate paper replay
launcher are under `archive/scripts/`.
The full notebook 09 recovery receipt is archived there because its diagnostic
sample includes original story and comment identifiers. The public
[investigation](../docs/notebook-09-recovery-investigation.md) reports the
aggregate findings and limitations.

The root `.gitignore` explicitly allows the public files above. Review any
new record for private paths and data before adding it to that list.
