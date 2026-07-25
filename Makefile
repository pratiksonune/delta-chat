.PHONY: setup setup-local-llm sample run chat ask markup eval test clean

PYTHON ?= python3
PAIR_DIR = data/samples/pair_001
OUT_DIR = out/pair_001
LLM_PROVIDER ?= mock

setup:
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages

setup-local-llm:
	$(PYTHON) -m pip install -r requirements-local-llm.txt --break-system-packages

sample:
	$(PYTHON) $(PAIR_DIR)/make_pid_b.py

run:
	$(PYTHON) run.py pipeline --pid-a $(PAIR_DIR)/pid_a.pdf --pid-b $(PAIR_DIR)/pid_b.pdf --out $(OUT_DIR)

chat:
	LLM_PROVIDER=$(LLM_PROVIDER) $(PYTHON) run.py chat --state $(OUT_DIR)/state.pkl

ask:
	$(PYTHON) run.py chat --state $(OUT_DIR)/state.pkl --question "$(Q)"

markup:
	$(PYTHON) run.py markup --state $(OUT_DIR)/state.pkl --out $(OUT_DIR)/markup

eval:
	$(PYTHON) eval/run_eval.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf out observability_runs
