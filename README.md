# Back of the Neural Net — More than a Hack 2027

Rule-based possession labelling for the **BAR–ATH** match (More than a Hack 2027, Metrica Sports).
Every video frame gets a label: `H` / `A` (home / away in possession), `D` (dead ball) or `X` (ungraded).

- **Approach:** sticky nearest-player possession, plus five dead-ball detectors (throw-ins, goals, goal kicks, corners, fouls / free kicks). No trained model.
- **Entry status:** solo entry, submitted for evaluation only. See the technical document.

📄 **Technical document:** [`back-of-the-neural-net.md`](back-of-the-neural-net.md)

## Repository layout

```
make_submission.py          # the full pipeline: data in, submission CSV out
back-of-the-neural-net.md   # technical document
games.py                    # organiser helper (match metadata, graded halves), MIT
load_smart_tagging.py       # organiser helper (Smart Tagging events), MIT
requirements.txt            # pinned dependencies
data/                       # NOT included: put the match data bundle here
```

## Reproduce the submission

```bash
git clone https://github.com/nicolascrifasi1603/back-of-the-neural-net.git
cd back-of-the-neural-net
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the match data bundle in `data/`. The data is licensed for this competition only and is not included.

```bash
python make_submission.py          # writes back-of-the-neural-net.csv
```

To check the file, use the organisers' validator from the [challenge repo](https://github.com/metrica-sports/more-than-a-hack-2027):
`python evaluation/validate_submission.py back-of-the-neural-net.csv`

The pipeline is deterministic (no randomness, no training), so this reproduces the submitted file exactly.

## Licence

Code and documentation: MIT (see `LICENSE`). The match data is not covered and is not redistributable.
