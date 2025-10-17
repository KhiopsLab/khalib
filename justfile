alias pc := pre-commit
pre-commit:
  uv run pre-commit run --all-files

alias d := docs
docs:
  (cd docs && rm -rf _build && uv run sphinx-build --nitpicky . _build/html)

alias t := test
[positional-arguments]
@test *args:
  uv run pytest src/tests "$@"


