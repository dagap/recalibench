EXP    ?= demo
MODE   ?= mock
TASKS  ?=                      # optional override, e.g. TASKS=osha_amputation
MODELS ?=                      # optional override, e.g. MODELS=jev,tfidf_logreg
LIMIT  ?= 20                   # items per task for make smoke
RUN     = uv run recal
ARGS    = experiment=$(EXP) mode=$(MODE) \
          $(if $(TASKS),experiment.tasks=[$(TASKS)]) \
          $(if $(MODELS),experiment.models=[$(MODELS)])

.PHONY: all list download prepare estimate smoke score calibrate evaluate experiments report demo test lint clean

all:         ; $(RUN) run $(ARGS)
list:        ; $(RUN) list
download:    ; $(RUN) download $(ARGS)
prepare:     ; $(RUN) prepare $(ARGS)
estimate:    ; $(RUN) score $(ARGS) dry_run=true
smoke:       ; $(RUN) score $(subst mode=$(MODE),mode=live,$(ARGS)) limit=$(LIMIT)   # live, LIMIT items per task, no artifacts
score:       ; $(RUN) score $(ARGS)
calibrate:   ; $(RUN) calibrate $(ARGS)
evaluate:    ; $(RUN) evaluate $(ARGS)
experiments: ; $(RUN) experiments $(ARGS)
report:      ; $(RUN) report $(ARGS)
demo:        ; uv run streamlit run app/streamlit_app.py

test:  ; uv run pytest -q
lint:  ; uv run ruff check . && uv run pyright && uv run deptry .
clean: ; rm -rf artifacts_mock reports data/processed   # keeps raw data, the cache and live artifacts
