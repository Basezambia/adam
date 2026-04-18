"""
live_learning.py — feed ADAM a URL and watch it learn, token by token.

Pipeline per URL:
  1. Fetch HTML / PDF / arXiv abstract
  2. Clean → plain text
  3. Split into ~512-token chunks
  4. For each chunk: ContinualLearner.wake_tick(chunk)
     → EWC-anchored tiny update + novelty-fracture gating
  5. Every tick emits an event: {loss, tension, fracture, z, chunk_idx, url}
     → surfaced via /learn/stream SSE

Safety rails inherited from ContinualLearner:
  - embeddings / LM head / memory slots frozen
  - grad clipped, per-step LR tiny (5e-7 / 5e-6 on fracture)
  - EWC anchor to pre-training weights → no catastrophic forgetting
"""
from __future__ import annotations
import re
import time
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from collections import deque

import requests
from bs4 import BeautifulSoup


# ─── Web fetchers ──────────────────────────────────────────────────────

ARXIV_RE = re.compile(r'^(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?$', re.IGNORECASE)
WIKI_RE  = re.compile(r'^wiki:(.+)$', re.IGNORECASE)

_HEADERS = {
    'User-Agent': 'ADAM-LiveLearning/0.6 (research; contact: lord-magus@hexq)'
}


def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Strip noise
    for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                     'aside', 'form', 'iframe', 'noscript']):
        tag.decompose()
    # Prefer <main> or <article> if available
    main = soup.find('main') or soup.find('article') or soup.body or soup
    text = main.get_text(separator=' ', strip=True)
    # Collapse whitespace and drop insanely short fragments
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fetch_arxiv(arxiv_id: str) -> Dict:
    """Use arXiv's Atom API → title + abstract (high-signal, no cruft)."""
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    r = requests.get(url, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'xml')
    entry = soup.find('entry')
    if not entry:
        raise RuntimeError(f'arXiv id {arxiv_id} not found')
    title = entry.title.get_text(strip=True) if entry.title else ''
    abs_  = entry.summary.get_text(strip=True) if entry.summary else ''
    authors = [a.find('name').get_text(strip=True)
               for a in entry.find_all('author') if a.find('name')]
    return {
        'source': f'arxiv:{arxiv_id}',
        'title':  title,
        'authors': authors,
        'text':   f'{title}\n\nAuthors: {", ".join(authors)}\n\n{abs_}',
        'url':    f'https://arxiv.org/abs/{arxiv_id}',
    }


def _fetch_wikipedia(title: str) -> Dict:
    """Fetch the FULL Wikipedia article (not just the summary) so live
    learning has enough chunks to be interesting. Falls back to REST
    summary on error."""
    t = title.strip().replace(' ', '_')
    full_url = f'https://en.wikipedia.org/wiki/{t}'
    try:
        r = requests.get(full_url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        return {
            'source': f'wiki:{title}',
            'title':  title.replace('_', ' '),
            'text':   _clean_html(r.text),
            'url':    full_url,
        }
    except Exception:
        # Fallback to summary
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{t}'
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {
            'source': f'wiki:{title}',
            'title':  data.get('title', title),
            'text':   data.get('extract', ''),
            'url':    data.get('content_urls', {}).get('desktop', {}).get('page', ''),
        }


def _fetch_generic(url: str) -> Dict:
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    ctype = r.headers.get('content-type', '').lower()
    if 'pdf' in ctype:
        # Cheap PDF: rely on pdfminer if installed; else fall back to empty
        try:
            from io import BytesIO
            from pdfminer.high_level import extract_text
            txt = extract_text(BytesIO(r.content))
        except Exception:
            txt = ''
        return {'source': url, 'title': url, 'text': txt, 'url': url}
    return {'source': url, 'title': url, 'text': _clean_html(r.text), 'url': url}


def fetch(target: str) -> Dict:
    """Dispatch: arxiv:<id>, wiki:<title>, or a plain URL."""
    target = target.strip()
    m = ARXIV_RE.match(target)
    if m:
        return _fetch_arxiv(m.group(1))
    m = WIKI_RE.match(target)
    if m:
        return _fetch_wikipedia(m.group(1))
    if not target.startswith(('http://', 'https://')):
        target = 'https://' + target
    return _fetch_generic(target)


# ─── Chunker ───────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_chars: int = 1600,
               min_chars: int = 120) -> List[str]:
    """Sentence-aware chunking. ~1600 chars ≈ 400-500 GPT-2 tokens."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chars else []
    # Split on sentence-ish boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], ''
    for s in sentences:
        if len(buf) + len(s) + 1 > chunk_chars and len(buf) >= min_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + ' ' + s).strip()
    if len(buf) >= min_chars:
        chunks.append(buf.strip())
    return chunks


# ─── Live learning engine ──────────────────────────────────────────────

@dataclass
class LearnJob:
    job_id: int
    target: str
    status: str = 'queued'    # queued | fetching | learning | done | error
    error: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    total_chunks: int = 0
    chunks_done: int = 0
    loss_history: List[float] = field(default_factory=list)
    fracture_events: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0


class LiveLearner:
    """Background thread that consumes a URL queue and streams wake_tick
    events to any subscribed listeners.

    The underlying learning is just `ContinualLearner.wake_tick(chunk)` —
    no new math, but now driven by arbitrary web content on demand.
    """

    def __init__(self, continual_learner, buffer=None,
                 tick_delay_s: float = 0.05):
        self.cl = continual_learner
        self.buf = buffer
        self.tick_delay_s = tick_delay_s
        self._q: queue.Queue[LearnJob] = queue.Queue()
        self._jobs: Dict[int, LearnJob] = {}
        self._next_id = 1
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._running_job_id: Optional[int] = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self.events: deque = deque(maxlen=500)

    # ── public API ──

    def submit(self, target: str) -> LearnJob:
        with self._lock:
            jid = self._next_id
            self._next_id += 1
            job = LearnJob(job_id=jid, target=target)
            self._jobs[jid] = job
            self._q.put(job)
        self._emit({'type': 'queued', 'job_id': jid, 'target': target})
        return job

    def pause(self):  self._paused.set()
    def resume(self): self._paused.clear()
    def stop_current(self):
        # Drop the current job by marking it cancelled; worker checks.
        jid = self._running_job_id
        if jid and jid in self._jobs:
            self._jobs[jid].status = 'cancelled'

    def jobs(self) -> List[Dict]:
        with self._lock:
            return [self._as_dict(j) for j in self._jobs.values()]

    def job(self, jid: int) -> Optional[Dict]:
        with self._lock:
            j = self._jobs.get(jid)
            return self._as_dict(j) if j else None

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=500)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        if q in self._subs:
            self._subs.remove(q)

    def stats(self) -> Dict:
        return {
            'queued':   self._q.qsize(),
            'jobs':     len(self._jobs),
            'paused':   self._paused.is_set(),
            'running':  self._running_job_id,
            'subs':     len(self._subs),
            'updates_applied': getattr(self.cl, 'updates_applied', 0),
            'fracture_updates': getattr(self.cl, 'fracture_updates', 0),
            'recent_losses': list(self.cl.loss_history)[-50:]
                if hasattr(self.cl, 'loss_history') else [],
        }

    # ── internals ──

    @staticmethod
    def _as_dict(j: LearnJob) -> Dict:
        return {
            'job_id': j.job_id, 'target': j.target, 'status': j.status,
            'error': j.error, 'url': j.url, 'title': j.title,
            'total_chunks': j.total_chunks, 'chunks_done': j.chunks_done,
            'loss_history': j.loss_history[-100:],
            'fracture_events': j.fracture_events,
            'started_at': j.started_at, 'finished_at': j.finished_at,
        }

    def _emit(self, event: Dict):
        event['t'] = time.time()
        self.events.append(event)
        dead = []
        for q in self._subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try: self._subs.remove(q)
            except ValueError: pass

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._process(job)

    def _process(self, job: LearnJob):
        self._running_job_id = job.job_id
        try:
            # 1. fetch
            job.status = 'fetching'
            job.started_at = time.time()
            self._emit({'type': 'fetching', 'job_id': job.job_id,
                        'target': job.target})
            doc = fetch(job.target)
            job.url = doc.get('url')
            job.title = doc.get('title') or job.target
            chunks = chunk_text(doc['text'])
            job.total_chunks = len(chunks)
            self._emit({'type': 'fetched', 'job_id': job.job_id,
                        'url': job.url, 'title': job.title,
                        'chunks': len(chunks),
                        'chars': len(doc['text'])})
            if not chunks:
                job.status = 'done'
                job.error = 'empty document after cleaning'
                job.finished_at = time.time()
                self._emit({'type': 'done', 'job_id': job.job_id,
                            'error': job.error})
                return
            # 2. learn
            job.status = 'learning'
            for i, chunk in enumerate(chunks):
                while self._paused.is_set():
                    time.sleep(0.1)
                if job.status == 'cancelled':
                    self._emit({'type': 'cancelled', 'job_id': job.job_id})
                    return
                res = self.cl.wake_tick(chunk)
                # Log to experience buffer too (so sleep_consolidate can replay)
                if self.buf is not None and res is not None:
                    try:
                        self.buf.add(text=chunk,
                                     tension=float(res.get('z', 0.0)),
                                     tag='web',
                                     fracture=bool(res.get('fracture', False)))
                    except Exception:
                        pass
                job.chunks_done = i + 1
                if res is not None:
                    job.loss_history.append(res['loss'])
                    if res.get('fracture'):
                        job.fracture_events += 1
                    self._emit({
                        'type': 'tick',
                        'job_id': job.job_id,
                        'chunk_idx': i,
                        'total_chunks': len(chunks),
                        'loss': res['loss'],
                        'fracture': bool(res.get('fracture', False)),
                        'z': res.get('z', 0.0),
                        'lr': res.get('lr_used', 0.0),
                        'text_preview': chunk[:120],
                    })
                time.sleep(self.tick_delay_s)
            job.status = 'done'
            job.finished_at = time.time()
            self._emit({'type': 'done', 'job_id': job.job_id,
                        'chunks_done': job.chunks_done,
                        'fracture_events': job.fracture_events,
                        'final_loss': (job.loss_history[-1]
                                       if job.loss_history else None)})
        except Exception as e:
            job.status = 'error'
            job.error = str(e)
            job.finished_at = time.time()
            self._emit({'type': 'error', 'job_id': job.job_id,
                        'error': str(e)})
        finally:
            self._running_job_id = None


# ─── quick self-test ───────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    tgt = sys.argv[1] if len(sys.argv) > 1 else 'wiki:Sudoku'
    doc = fetch(tgt)
    print(f"[{doc['source']}] {doc['title']}")
    print(f"  url:   {doc['url']}")
    print(f"  chars: {len(doc['text'])}")
    chunks = chunk_text(doc['text'])
    print(f"  chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"  [{i}] {c[:120]}...")
