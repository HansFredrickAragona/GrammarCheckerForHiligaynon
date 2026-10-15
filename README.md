# Hiligaynon Grammar Checker

A Streamlit application for checking simple Hiligaynon sentence patterns, exploring the vocabulary used by the grammar, and searching the included dictionaries.

## Run the app

Install the project dependencies, then start Streamlit from this folder:

```powershell
pip install -r requirements.txt
streamlit run hiligaynon_checker/app.py
```

## Project layout

```text
hiligaynon_checker/       Streamlit application package
  app.py                  User interface
  grammar_engine/         Grammar rules, parser, and parse-tree helpers
  helpers/                Lexicon loading, word categories, and lookup helpers
data/
  dictionaries/           PinoyDictionary and Cecile Motus CSV dictionaries
  lexicons/               Active, raw, preprocessed, and sample lexicon files
  source_material/        Reference books and source notes
  corpora/                Language corpora and collected comments
  sentiment/              Sentiment lexicon files
  validation/             Respondent review and validation data
tools/                    Offline scraping and lexicon-processing scripts
  scraping/
  preprocessing/
  sampling/
  sentiment/
  analysis/
notebooks/                Exploratory notebooks
legacy/                   Original standalone grammar prototype
docs/                     Project notes and preserved source-project metadata
```

The app reads its active data from `data/lexicons/Ceciel_Wiki_merged_scores.csv` and the two CSVs in `data/dictionaries/`. The processing scripts write their outputs under `data/` so the interface and research data remain easy to find.
