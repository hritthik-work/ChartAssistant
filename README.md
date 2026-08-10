# HealthBot

HealthBot is a grounded healthcare-document assistant prototype.

The `main` branch contains the minimal application scaffold. The synthetic
healthcare corpus, curation plan, source requirements, and mock ICD-10 reference
data live on the `data_curate` branch.

## Project Layout

```text
src/healthbot/   Application package
tests/           Automated tests as the application is implemented
```

## Development

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

The package currently contains only the project scaffold. Application components
will be added in later branches.
