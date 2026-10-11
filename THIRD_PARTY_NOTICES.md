# Third-party notices

Original ExactKV material is licensed as described in [LICENSING.md](LICENSING.md).
Upstream benchmark text and models retain their original terms.

## CPython benchmark excerpts

The code prompts derive from the Python Software Foundation's CPython source,
revision `7352b6aaa99ed68b055cb3c063657a2e3ba82190`. Their upstream files and
hashes are recorded in [data/sources.json](data/sources.json), and prompt
records in [data/manifest.jsonl](data/manifest.jsonl). The complete upstream
license and historical notices are retained in
[LICENSES/CPython-LICENSE.txt](LICENSES/CPython-LICENSE.txt).

Source: <https://github.com/python/cpython/tree/7352b6aaa99ed68b055cb3c063657a2e3ba82190>.
The prompts are token-selected excerpts; they are not original ExactKV prose
and are not relicensed under CC BY 4.0.

## WikiText benchmark excerpts

Prose prompts derive from Salesforce WikiText-2, a dataset of Wikipedia text,
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, under CC BY-SA 3.0 as
recorded in [data/sources.json](data/sources.json). Credit the contributing
Wikipedia authors and the WikiText dataset creators. Dataset source:
<https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3>.

The prompts are token-selected excerpts. Original dataset file hashes and
split provenance remain in the source lock and manifest. The applicable
license is preserved in [LICENSES/CC-BY-SA-3.0.txt](LICENSES/CC-BY-SA-3.0.txt);
share-alike obligations for covered adaptations remain in force. These
excerpts are not relicensed under ExactKV's CC BY 4.0 grant.

## Model and dependencies

The Qwen3-0.6B checkpoint retains its upstream terms. Its exact revision and
file hashes are in [data/model-lock.json](data/model-lock.json). The model
and downloaded source datasets are not distributed as binary artifacts.
MLX, MLX-LM, NumPy, Zstandard, Hugging Face libraries, and all other
dependencies retain their own licenses and notices.
