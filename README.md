# NLP course project (private)

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

1. On GitHub, open this repository.
2. Go to **Settings** → **Collaborators and teams** (or **Manage access**).
3. Click **Invite a collaborator**, enter each teammate’s **GitHub username** (or email if offered), and choose **Write** or **Maintain** as appropriate.
4. Teammates must **accept the email/GitHub invitation** before they can push.

Optional: under **Settings → Branches**, add a branch protection rule for `main` (e.g. require pull requests before merging).
