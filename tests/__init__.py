# Regular package marker. Required: ultralytics >= 8.4 installs a top-level
# ``tests`` package into site-packages; without this __init__.py our tests/
# directory is only a PEP-420 namespace candidate and loses the import race,
# breaking every ``from tests.fixtures import ...`` in the suite.
