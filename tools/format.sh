for f in *.py; do docformatter --in-place $f; done
isort --skip-gitignore .
black src
black tests
