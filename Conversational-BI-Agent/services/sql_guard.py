import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"SALES", "REGIONS", "PRODUCTS"}
ALLOWED_VERBS = {"SELECT", "WITH"}

def validate_and_fix(sql: str) -> str:
    try:
        tree = sqlglot.parse_one(sql, read="snowflake")
    except Exception as e:
        raise ValueError(f"Invalid SQL: {e}")

    # Ensure verb is allowed
    if tree.key and tree.key.upper() not in ALLOWED_VERBS:
        raise ValueError("Only SELECT/WITH allowed.")

    # Ensure only allowed tables
    for t in tree.find_all(exp.Table):
        name = (t.name or "").upper()
        if name not in ALLOWED_TABLES:
            raise ValueError(f"Table not allowed: {name}")

    # Inject LIMIT when appropriate (no GROUP and no LIMIT)
    has_group = any(isinstance(e, exp.Group) for e in tree.find_all(exp.Group))
    has_limit = any(isinstance(e, exp.Limit) for e in tree.find_all(exp.Limit))
    if (not has_group) and (not has_limit):
        tree = tree.limit(200)

    return tree.sql(dialect="snowflake")