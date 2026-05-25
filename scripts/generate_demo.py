#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
generate_demo.py — Product demo video for the moonlight translation workbench.

Usage:
    python3 scripts/generate_demo.py
    python3 scripts/generate_demo.py --url http://localhost:8765 --out demo/demo.mp4
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

# ── Demo config ──────────────────────────────────────────────────────────────
DEMO_SENTENCE = (
    "The President of the Maldives signed the new Foreign Investment Act "
    "at the People's Majlis ceremony today."
)
GLOSSARY_SEARCH_EN = "foreign invest"
GLOSSARY_SEARCH_DV = "ރައީސ"


# ── Helpers ──────────────────────────────────────────────────────────────────

def pause(s: float) -> None:
    time.sleep(s)


def caption(page: Page, text: str, hold: float = 0.0) -> None:
    """Update the injected caption bar text."""
    page.evaluate(
        "t => { const el = document.getElementById('__demo_caption'); if(el) el.textContent = t; }",
        text,
    )
    if hold > 0:
        pause(hold)


def inject_caption_bar(page: Page) -> None:
    """Inject a persistent caption overlay at the bottom of the page."""
    page.evaluate("""() => {
        const bar = document.createElement('div');
        bar.id = '__demo_caption';
        bar.style.cssText = `
            position: fixed;
            bottom: 0; left: 0; right: 0;
            background: rgba(6, 8, 16, 0.94);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-top: 3px solid rgba(108, 142, 247, 0.7);
            padding: 18px 64px 22px;
            font: 700 28px/1.3 -apple-system, "SF Pro Display", system-ui, sans-serif;
            color: #ffffff;
            letter-spacing: 0.015em;
            z-index: 999999;
            text-align: center;
            pointer-events: none;
            text-shadow: 0 1px 12px rgba(0,0,0,0.8);
        `;
        document.body.appendChild(bar);
    }""")


def slow_type(page: Page, selector: str, text: str, delay_ms: int = 48) -> None:
    page.click(selector)
    page.type(selector, text, delay=delay_ms)


def click_tab(page: Page, label: str) -> None:
    try:
        page.get_by_role("tab", name=label).first.click()
    except Exception:
        page.locator(f".wb-tab:has-text('{label}')").first.click()
    pause(0.5)


def wait_for_translation(page: Page, timeout: int = 45_000) -> None:
    page.wait_for_selector("button:not([disabled]):has-text('Translate')", timeout=timeout)


def wait_for_alignment(page: Page, timeout: int = 20_000) -> None:
    """Wait until the Alignment tab loading dot disappears."""
    try:
        page.wait_for_function(
            """() => {
                for (const tab of document.querySelectorAll('.wb-tab')) {
                    if (tab.textContent.includes('Alignment')) {
                        const dot = tab.querySelector('span');
                        if (!dot) return true;
                        return window.getComputedStyle(dot).display === 'none';
                    }
                }
                return true;
            }""",
            timeout=timeout,
        )
    except Exception:
        pass


def scroll_to(page: Page, selector: str) -> None:
    try:
        page.eval_on_selector(selector, "el => el.scrollIntoView({behavior:'smooth',block:'nearest'})")
    except Exception:
        pass
    pause(0.4)


def scroll_to_panel(page: Page) -> None:
    """After clicking a tab, scroll the window so the panel content is visible."""
    page.evaluate("""() => {
        const tabs = document.querySelector('.wb-tabs');
        if (!tabs) return;
        // Scroll so the tab bar sits near the top — reveals the full panel below
        const target = tabs.getBoundingClientRect().top + window.scrollY - 10;
        window.scrollTo({ top: target, behavior: 'smooth' });
    }""")
    pause(0.7)


def hover_token(page: Page, word: str) -> None:
    try:
        page.locator(f".dv-token:has-text('{word}')").first.hover(timeout=2500)
    except Exception:
        pass


def click_token(page: Page, word: str) -> None:
    try:
        page.locator(f".dv-token:has-text('{word}')").first.click(timeout=2500)
    except Exception:
        pass


def hover_svg_text(page: Page, label: str) -> None:
    try:
        page.locator(f"text={label}").first.hover(timeout=2500)
    except Exception:
        pass


# ── Cache pre-warm ────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def pre_warm(base_url: str) -> str:
    """Pre-warm translation + alignment caches. Returns cached translation."""
    print("Pre-warming translation…", end=" ", flush=True)
    translation = ""
    try:
        data = _post_json(f"{base_url}/api/translate", {
            "text": DEMO_SENTENCE, "target_language": "DV",
            "mode": "faithful", "model": "sonnet",
        })
        translation = data.get("translation", "")
        print(f"done ({translation[:35]}…)")
    except Exception as e:
        print(f"WARN: {e}")
        return translation

    print("Pre-warming alignment…", end=" ", flush=True)
    try:
        _post_json(f"{base_url}/api/align-batch", {
            "source": DEMO_SENTENCE, "translation": translation,
            "source_lang": "EN", "target_lang": "DV",
        })
        print("done")
    except Exception as e:
        print(f"WARN: {e}")

    return translation


# ── Demo sequence ─────────────────────────────────────────────────────────────

def run_demo(page: Page, base_url: str) -> None:

    # ── TITLE CARD ────────────────────────────────────────────────────────────
    page.goto(f"{base_url}/workbench")
    page.wait_for_load_state("networkidle")
    inject_caption_bar(page)
    caption(page, "Moonlight · EN↔DV Translation Workbench", hold=2.0)

    # ── TYPE THE SENTENCE ─────────────────────────────────────────────────────
    caption(page, "Type any English sentence…")
    slow_type(page, "textarea[x-model='sourceText'], textarea", DEMO_SENTENCE, delay_ms=32)
    pause(0.4)

    # ── TRANSLATE ─────────────────────────────────────────────────────────────
    caption(page, "Claude Sonnet — glossary-grounded, entity-locked, register-aware")
    page.get_by_role("button", name="Translate").first.click()
    wait_for_translation(page)
    wait_for_alignment(page)
    pause(0.3)

    # ── SHOW OUTPUT ───────────────────────────────────────────────────────────
    caption(page, "Dhivehi — 95% confidence, entity check ✓, $0.01 cached")
    pause(1.8)

    # ── WORD BREAKDOWN ROW ────────────────────────────────────────────────────
    # Scroll the word-breakdown row into the centre of the frame
    page.evaluate("""() => {
        const wb = document.querySelector('[class*="word-breakdown"], .wb-breakdown, #wordBreakdown');
        if (wb) wb.scrollIntoView({behavior:'smooth', block:'center'});
        else window.scrollBy({top: 140, behavior:'smooth'});
    }""")
    pause(0.6)
    caption(page, "11 tokens — colour-coded: corpus-locked · entity · formal · honorific")
    pause(2.2)

    # ── BIDIRECTIONAL TOKEN HOVER ─────────────────────────────────────────────
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    pause(0.4)
    caption(page, "Hover any token — the matching English source word lights up")
    hover_token(page, "ރައީސ")       # President
    pause(1.8)
    page.mouse.move(400, 180)
    pause(0.3)

    # ── WORD DETAIL — click token while output is still visible ───────────────
    caption(page, "Click any token → dictionary entry + corpus concordance")
    click_token(page, "ރައްޔިތ")     # People's
    pause(0.3)
    click_tab(page, "Word Detail")
    scroll_to_panel(page)
    caption(page, "Glossary entry for 'ރައްޔިތ' — source alignment + register notes")
    pause(2.2)
    page.evaluate("window.scrollBy({top: 180, behavior: 'smooth'})")
    caption(page, "5 concordance snippets from the parallel EN↔DV corpus")
    pause(2.2)

    # ── ALIGNMENT TAB ────────────────────────────────────────────────────────
    caption(page, "Alignment arcs — every Dhivehi token mapped to its English source")
    click_tab(page, "Alignment")
    scroll_to_panel(page)
    page.evaluate("window.scrollBy({top: 80, behavior: 'smooth'})")
    pause(1.0)
    caption(page, "Hover a source word — its arcs illuminate, all others dim")
    hover_svg_text(page, "President")
    pause(2.5)
    hover_svg_text(page, "Investment")
    pause(2.0)
    hover_svg_text(page, "signed")
    pause(1.8)
    page.mouse.move(500, 300)
    pause(0.4)

    # ── NEURAL NET TAB ────────────────────────────────────────────────────────
    caption(page, "Neural Net — source tokens · NER entities · register nodes")
    click_tab(page, "Neural Net")
    scroll_to_panel(page)
    pause(2.5)
    try:
        canvas = page.locator("#neuralNetCanvas")
        box = canvas.bounding_box()
        if box:
            cx = box["x"] + box["width"] * 0.50
            caption(page, "Named entity 'LOC — Maldives' — hover to trace connections")
            page.mouse.move(cx, box["y"] + box["height"] * 0.25)
            pause(2.0)
            page.mouse.move(cx, box["y"] + box["height"] * 0.55)
            caption(page, "Drag any node — physics engine repositions in real time")
            pause(1.5)
            sx = box["x"] + box["width"] * 0.20
            sy = box["y"] + box["height"] * 0.40
            page.mouse.move(sx, sy); pause(0.2)
            page.mouse.down()
            page.mouse.move(sx + 55, sy - 35, steps=18); pause(0.3)
            page.mouse.up()
            pause(1.4)
    except Exception:
        pause(2.5)

    # ── GLOSSARY TAB ─────────────────────────────────────────────────────────
    caption(page, "Glossary — 3,600+ verified EN↔DV term pairs from the corpus")
    click_tab(page, "Glossary")
    scroll_to_panel(page)
    pause(0.8)
    try:
        search = page.locator(".glossary-search-input").first
        search.click()
        caption(page, "Search in English…")
        search.fill(GLOSSARY_SEARCH_EN)
        pause(2.0)
        page.evaluate("window.scrollBy({top: 120, behavior: 'smooth'})")
        pause(0.4)
        search.fill("")
        pause(0.15)
        caption(page, "…or type directly in Dhivehi script")
        search.type(GLOSSARY_SEARCH_DV, delay=70)
        pause(2.0)
        page.evaluate("window.scrollBy({top: 120, behavior: 'smooth'})")
        pause(0.4)
        search.fill("")
        pause(0.15)
    except Exception:
        pause(3.5)

    # ── BENCHMARKS TAB ───────────────────────────────────────────────────────
    caption(page, "Benchmarks — Moonlight leads DhivehiMT-Bench vs GPT-4o, Gemini")
    click_tab(page, "Benchmarks")
    scroll_to_panel(page)
    pause(0.8)
    page.evaluate("window.scrollBy({top: 180, behavior: 'smooth'})")
    pause(0.5)
    try:
        page.locator(".wb-table, table").last.scroll_into_view_if_needed()
        caption(page, "chrF++ 49.3 · BLEU 22.4 · humans rated Moonlight output best")
        pause(2.2)
    except Exception:
        pause(2.0)

    # ── CLOSING ───────────────────────────────────────────────────────────────
    page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
    pause(0.5)
    caption(page, "moonlight-workbench — open-source EN↔DV translation engine")
    pause(3.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",    default="http://localhost:8765")
    ap.add_argument("--out",    default="demo/moonlight_workbench_demo.mp4")
    ap.add_argument("--width",  type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ {out}  ({args.width}×{args.height})")
    pre_warm(args.url)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(out.parent),
            record_video_size={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        try:
            run_demo(page, args.url)
        finally:
            ctx.close()
            browser.close()

    videos = sorted(out.parent.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if videos:
        webm = videos[-1]
        webm.rename(out.with_suffix(".webm"))
        print(f"\n✓  {out.with_suffix('.webm')}")
        print(f"   ffmpeg -i {out.with_suffix('.webm')} -c:v libx264 -pix_fmt yuv420p {out}")
    else:
        print("⚠  no .webm found")


if __name__ == "__main__":
    main()
