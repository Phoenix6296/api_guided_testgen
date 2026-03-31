import json
from pathlib import Path

root = Path(__file__).resolve().parent
data_dir = root / "data"
api_db_data = data_dir / "api_db"
api_db_top = root / "api_db"

if not api_db_data.is_dir():
    raise FileNotFoundError(
        f"Required directory does not exist: {api_db_data}. Create it before running this script."
    )
if not api_db_top.is_dir():
    raise FileNotFoundError(
        f"Required directory does not exist: {api_db_top}. Create it before running this script."
    )

libs = ["tf", "torch", "sklearn", "xgb", "jax"]

def read_lines(path: Path):
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

all_apidoc_docs = []
for lib in libs:
    apis = read_lines(data_dir / f"{lib}_api_list.txt")
    links = read_lines(data_dir / f"{lib}_apidoc_link.txt")

    write_jsonl(api_db_data / f"api_class_over_10_{lib}.jsonl", [{"api": a, "paths": []} for a in apis])

    apidoc_rows = []
    for i, a in enumerate(apis):
        apidoc_rows.append({
            "title": a,
            "signature": "",
            "nl_descs": "",
            "ex_codes": "",
            "doc_url": links[i] if i < len(links) else "",
        })
        all_apidoc_docs.append({"title": a, "document": "Signature: \nDescriptions: \nExample code:"})
    write_jsonl(api_db_data / f"apidoc_{lib}.jsonl", apidoc_rows)

    for src in ["issues", "sos", "repos"]:
        write_jsonl(api_db_data / f"{src}_{lib}.jsonl", [])

    write_jsonl(
        api_db_data / f"sorted_{lib}_over10_new.jsonl",
        [{"api_name": a, "issues": [], "sos": [], "repos": []} for a in apis],
    )

write_jsonl(api_db_data / "basic_rag_apidoc.jsonl", all_apidoc_docs)
for name in ["basic_rag_issues", "basic_rag_sos", "basic_rag_repos", "basic_rag_all", "similarity"]:
    write_jsonl(api_db_data / f"{name}.jsonl", all_apidoc_docs)

for p in api_db_top.glob("*.jsonl"):
    p.unlink()
for p in api_db_data.glob("*.jsonl"):
    (api_db_top / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

print("rebuilt")
for lib in libs:
    c = sum(1 for _ in (api_db_data / f"api_class_over_10_{lib}.jsonl").open("r", encoding="utf-8"))
    print(lib, c)
