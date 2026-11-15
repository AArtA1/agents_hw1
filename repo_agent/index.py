"""Repository index: BM25 over code chunks and commits + a symbol table.

Pure Python (no vector DB), persisted as JSON under workspace/<slug>/index/.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import config, issues as issues_mod, repo as repo_mod
from .chunking import Chunk, Symbol, chunk_file, extract_symbols
from .textutil import is_aux_path, is_test_path, split_lines

_STOP = set("""a an the of to in on for and or is are was were be been it its this that these those
with as by at from into how what where which who why when does do did can could should would
there here then than so if not no yes i you we they he she my our your their me us them about
используется где как что какой какая какие почему зачем это в на и или ли для из по""".split())

_SPLIT = re.compile(r"[A-Za-z0-9]+|[а-яА-ЯёЁ]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Code-aware tokens: identifiers are kept whole *and* split by camelCase/snake_case."""
    out: list[str] = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[а-яА-ЯёЁ]+|\d+", text):
        low = ident.lower()
        parts = [p.lower() for piece in ident.split("_") if piece for p in _CAMEL.findall(piece)]
        if low not in _STOP and len(low) > 1:
            out.append(low)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in _STOP and len(p) > 1)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                df[t] += 1
                self.postings[t].append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> dict[int, float]:
        res: dict[int, float] = defaultdict(float)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                res[i] += idf * f * (self.k1 + 1) / denom
        return res


@dataclass
class Hit:
    chunk: Chunk
    score: float


class RepoIndex:
    """Everything the agent knows about one repository."""

    def __init__(self, root: Path, slug: str, chunks: list[Chunk], symbols: list[Symbol],
                 commits: list[repo_mod.Commit], files: list[str], meta: dict):
        self.root, self.slug = root, slug
        self.chunks, self.symbols, self.commits, self.files, self.meta = chunks, symbols, commits, files, meta
        self._file_set = set(files)
        self._sym_by_name: dict[str, list[Symbol]] = defaultdict(list)
        for s in symbols:
            self._sym_by_name[s.name.lower()].append(s)
        self._bm25 = BM25([self._doc_tokens(c) for c in chunks])
        self._commit_bm25 = BM25([tokenize(f"{c.subject} {c.body} {' '.join(c.files)}") for c in commits])
        self.issues: list[dict] = issues_mod.load_issues(slug)
        self._issue_bm25 = BM25([tokenize(f"{i['title']} {i['title']} {i['body']}") for i in self.issues])

    # ------------------------------------------------------------ building
    @staticmethod
    def _doc_tokens(c: Chunk) -> list[str]:
        # path and defined symbols are strong signals -> weight them x3
        head = f"{c.path} {' '.join(c.symbols)}"
        return tokenize(head) * 3 + tokenize(c.text)

    @classmethod
    def build(cls, spec: str, depth: int | None = None, progress=None, with_issues: bool = True) -> "RepoIndex":
        t0 = time.time()
        root, slug = repo_mod.fetch(spec, depth=depth)
        files = repo_mod.list_files(root)
        chunks: list[Chunk] = []
        symbols: list[Symbol] = []
        for i, rel in enumerate(files):
            text = repo_mod.read_text(root, rel)
            if text is None:
                continue
            syms = extract_symbols(rel, text)
            symbols += syms
            chunks += chunk_file(rel, text, syms, len(chunks))
            if progress and i % 50 == 0:
                progress(i, len(files), rel)
        commits = repo_mod.commits(root)
        web_url = repo_mod.remote_web_url(root)
        issues_note = None
        if with_issues and web_url:
            try:  # best effort: the index is useful without issues
                issues_mod.fetch_issues(slug, web_url)
            except issues_mod.IssuesError as e:
                issues_note = str(e)
        meta = {
            "slug": slug, "spec": spec, "root": str(root), "head": repo_mod.head_sha(root),
            "web_url": repo_mod.remote_web_url(root), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_files": len(files), "n_chunks": len(chunks), "n_symbols": len(symbols),
            "n_commits": len(commits), "n_issues": len(issues_mod.load_issues(slug)),
            "issues_error": issues_note, "build_seconds": round(time.time() - t0, 2),
        }
        idx = cls(root, slug, chunks, symbols, commits, files, meta)
        idx.save()
        return idx

    # --------------------------------------------------------- persistence
    @staticmethod
    def index_dir(slug: str) -> Path:
        return config.WORKSPACE / slug / "index"

    def save(self) -> Path:
        d = self.index_dir(self.slug)
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))
        (d / "files.json").write_text(json.dumps(self.files))
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False))
        (d / "symbols.json").write_text(json.dumps([s.to_dict() for s in self.symbols], ensure_ascii=False))
        (d / "commits.json").write_text(json.dumps([c.__dict__ for c in self.commits], ensure_ascii=False))
        return d

    @classmethod
    def load(cls, slug: str) -> "RepoIndex":
        d = cls.index_dir(slug)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"Индекс для {slug} не найден. Сначала: repo-agent index <repo>")
        meta = json.loads((d / "meta.json").read_text())
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        symbols = [Symbol(**s) for s in json.loads((d / "symbols.json").read_text())]
        commits = [repo_mod.Commit(**c) for c in json.loads((d / "commits.json").read_text())]
        files = json.loads((d / "files.json").read_text())
        return cls(Path(meta["root"]), slug, chunks, symbols, commits, files, meta)

    @classmethod
    def load_or_build(cls, spec: str, rebuild: bool = False, **kw) -> "RepoIndex":
        _, slug = repo_mod.parse_repo_spec(spec)
        if not rebuild:
            try:
                return cls.load(slug)
            except FileNotFoundError:
                pass
        return cls.build(spec, **kw)

    @staticmethod
    def list_indexed() -> list[str]:
        if not config.WORKSPACE.exists():
            return []
        return sorted(p.parent.parent.name for p in config.WORKSPACE.glob("*/index/meta.json"))

    # -------------------------------------------------------------- search
    def search(self, query: str, k: int = 8, path_prefix: str | None = None,
               kinds: tuple[str, ...] | None = None) -> list[Hit]:
        q = tokenize(query)
        scores = self._bm25.scores(q)
        path_prefix = _norm_rel(path_prefix) if path_prefix else None
        # exact symbol-name boost
        for tok in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)):
            for s in self._sym_by_name.get(tok.lower(), []):
                for c in self.chunks_for(s.path, s.line, s.line):
                    scores[c.id] = scores.get(c.id, 0.0) + 5.0
        hits = []
        for cid, sc in scores.items():
            c = self.chunks[cid]
            if path_prefix and not c.path.startswith(path_prefix):
                continue
            if kinds and c.kind not in kinds:
                continue
            # light penalty for tests/docs/examples so implementation code ranks first
            if is_test_path(c.path) or is_aux_path(c.path):
                sc *= 0.8
            hits.append(Hit(c, sc))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    def search_files(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """File-level ranking (max chunk score per file)."""
        best: dict[str, float] = {}
        for h in self.search(query, k=60):
            best[h.chunk.path] = max(best.get(h.chunk.path, 0.0), h.score)
        return sorted(best.items(), key=lambda kv: -kv[1])[:k]

    def search_commits(self, query: str, k: int = 8) -> list[repo_mod.Commit]:
        scores = self._commit_bm25.scores(tokenize(query))
        return [self.commits[i] for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]

    def search_issues(self, query: str, k: int = 8) -> list[dict]:
        scores = self._issue_bm25.scores(tokenize(query))
        return [self.issues[i] for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]

    def chunks_for(self, path: str, start: int, end: int) -> list[Chunk]:
        return [c for c in self.chunks if c.path == path and c.start <= end and c.end >= start]

    def find_symbol(self, name: str) -> list[Symbol]:
        name = name.strip()
        if "." in name:
            return [s for s in self.symbols if s.qualname == name or s.qualname.endswith("." + name)]
        exact = self._sym_by_name.get(name.lower(), [])
        if exact:
            return exact
        low = name.lower()
        return [s for s in self.symbols if low in s.name.lower()][:30]

    # --------------------------------------------------------------- files
    def has_file(self, rel: str) -> bool:
        return rel in self._file_set

    def resolve_path(self, rel: str, context: list[str] | None = None) -> str | None:
        """Exact path, or a unique suffix ('utils.py' -> 'src/pkg/utils.py').

        Ambiguous suffixes are disambiguated by `context` (other paths mentioned nearby: the
        candidate sharing the longest directory prefix wins) and then by preferring non-test,
        non-docs files; if still ambiguous -> None.
        """
        rel = _norm_rel(rel)
        if not rel:
            return None
        if rel in self._file_set:
            return rel
        cands = [f for f in self.files if f.endswith("/" + rel)]
        if len(cands) <= 1:
            return cands[0] if cands else None
        if context:
            def shared(c: str) -> int:
                d = c.rsplit("/", 1)[0]
                return max((len(_common_dir(d, x)) for x in context if x != c), default=0)
            best = max(shared(c) for c in cands)
            top = [c for c in cands if shared(c) == best]
            if best and len(top) == 1:
                return top[0]
        prod = [c for c in cands if not is_test_path(c) and not is_aux_path(c)]
        return prod[0] if len(prod) == 1 else None

    def read_lines(self, rel: str, start: int = 1, end: int | None = None) -> tuple[list[str], int]:
        text = repo_mod.read_text(self.root, rel) or ""
        lines = split_lines(text)
        end = len(lines) if end is None else min(end, len(lines))
        start = max(1, start)
        return lines[start - 1:end], len(lines)

    def blob_url(self, rel: str, start: int | None = None, end: int | None = None) -> str | None:
        base = self.meta.get("web_url")
        if not base:
            return None
        url = f"{base}/blob/{self.meta.get('head', 'HEAD')}/{rel}"
        if start:
            url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
        return url


def _is_test(path: str) -> bool:  # kept for backwards compatibility
    return is_test_path(path)


def _norm_rel(rel: str) -> str:
    """'./src/x.py' -> 'src/x.py', '/a/b' -> 'a/b', '.' -> '' (removes the prefix, not characters)."""
    rel = rel.strip().replace("\\", "/")
    while rel.startswith(("./", "../")):
        rel = rel[2:] if rel.startswith("./") else rel[3:]
    rel = rel.lstrip("/")
    return "" if rel == "." else rel


def _common_dir(a: str, b: str) -> str:
    pa, pb = a.split("/"), b.split("/")
    out = []
    for x, y in zip(pa, pb):
        if x != y:
            break
        out.append(x)
    return "/".join(out)
