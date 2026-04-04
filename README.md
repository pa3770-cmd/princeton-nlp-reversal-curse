# NLP course project (private)

**Repository:** https://github.com/ashishkg0202/princeton-nlp-reversal-curse (private)

Reversal-curse / multi-hop dataset code and tests. Generated JSONL data under `code/data/` is not tracked; regenerate locally with the `generate_*.py` scripts in `code/`.

## Setup

```bash
cd code
pip install pytest
# Optional: UTF-8 on Windows when running generators
set PYTHONUTF8=1
py -3 generate_multihop.py
# ... other generators as needed
```

## Collaboration (invite teammates)

Only people you add can read or push to this **private** repository.

1. Go to **[Collaborators settings](https://github.com/ashishkg0202/princeton-nlp-reversal-curse/settings/access)** (repo **Settings** → **Collaborators and teams** / **Manage access**).
2. Click **Invite a collaborator**, enter each teammate’s **GitHub username** (or email if offered), and choose **Write** or **Maintain** as appropriate.
3. Teammates must **accept the invitation** before they can push.

Optional: under **Settings → Branches**, add a branch protection rule for `main` (e.g. require pull requests before merging).
