PYTHONPATH=src python - <<'PY'
from utils.canonical_key_mapper import get_canonical_mapper
mapper = get_canonical_mapper()
print(mapper.get_tags('total_debt', 'Technology'))
print(mapper.get_tags('short_term_debt', 'Technology'))
print(mapper.get_tags('long_term_debt', 'Technology'))
PY
