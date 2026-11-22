EXP    ?= demo
MODE   ?= mock
TASKS  ?=                      # optional override, e.g. TASKS=osha_amputation
MODELS ?=                      # optional override, e.g. MODELS=jev,tfidf_logreg
RUN     = uv run recal
ARGS    = experiment=$(EXP) mode=$(MODE) \
          $(if $(TASKS),experiment.tasks=[$(TASKS)]) \
          $(if $(MODELS),experiment.models=[$(MODELS)])

.PHONY: all list download prepare estimate score calibrate evaluate experiments report demo test lint clean

all:         ; $(RUN) run $(ARGS)
list:        ; $(RUN) list
download:    ; $(RUN) download $(ARGS)
prepare:     ; $(RUN) prepare $(ARGS)
estimate:    ; $(RUN) score $(ARGS) dry_run=true
score:       ; $(RUN) score $(ARGS)
calibrate:   ; $(RUN) calibrate $(ARGS)
evaluate:    ; $(RUN) evaluate $(ARGS)
experiments: ; $(RUN) experiments $(ARGS)
report:      ; $(RUN) report $(ARGS)
demo:        ; uv run streamlit run app/streamlit_app.py

test:  ; uv run pytest -q
lint:  ; uv run ruff check . && uv run pyright && uv run deptry .
clean: ; rm -rf artifacts_mock reports data/processed   # keeps raw data, the cache and live artifacts
