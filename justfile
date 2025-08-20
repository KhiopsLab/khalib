alias pc := pre-commit
pre-commit:
  uv run pre-commit run --all-files

alias t := test
[positional-arguments]
@test *args:
  uv run pytest src/tests "$@"
