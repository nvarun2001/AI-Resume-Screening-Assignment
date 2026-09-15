# AI-Resume-Screening-Assignment

AI resume screening and ranking pipeline for the SDE Intern assignment.

Ingests a folder of resumes, filters candidates against hard Python/AI
eligibility rules, scores eligible candidates on a 100-point rubric,
enriches scores with public GitHub activity, and writes a ranked
`results.json`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your LLM API key

python main.py --input ./resumes --output ./output/results.json
```

Full documentation (design decisions, scoring strategy) will be added as
the implementation lands.
