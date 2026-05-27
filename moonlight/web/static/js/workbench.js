// SPDX-License-Identifier: Apache-2.0
// Translation workbench — Alpine.js app

// ── Thaana tokenizer ─────────────────────────────────────────────────────────
// Split on whitespace; preserve tokens that contain at least one Thaana char
// or are purely Latin/numeric (for mixed EN/DV output).
function tokenizeDV(text) {
    if (!text) return [];
    return text.split(/\s+/).filter(t => t.length > 0).map(word => {
        const clean = word.replace(/[.،؟!،؟۔"']+$/, '');
        return { text: word, clean: clean || word, register: registerOf(clean || word) };
    });
}

// ── Register badge detection ─────────────────────────────────────────────────
// Check Thaana verb suffixes that encode politeness register.
// Ordered longest-first so honorific forms match before their substrings.
const _REGISTER = [
    { re: /ވިދާޅުވިއެވެ$/,  label: 'honorific',  cls: 'badge-purple' },
    { re: /ވިދާޅުވެއްޖެ$/,  label: 'perf-honor', cls: 'badge-yellow' },
    { re: /ވިއެވެ$/,         label: 'formal',     cls: 'badge-blue'   },
    { re: /ވެއްޖެ$/,         label: 'perfective', cls: 'badge-amber'  },
    { re: /ބުންޏެވެ$/,       label: 'informal',   cls: 'badge-dim'    },
];

function registerOf(token) {
    for (const { re, label, cls } of _REGISTER) {
        if (re.test(token)) return { label, cls };
    }
    return null;
}

// ── Thaana detection ─────────────────────────────────────────────────────────
function isThaana(text) {
    if (!text) return false;
    const nonWs = (text.match(/\S/g) || []).length;
    if (!nonWs) return false;
    const thaana = (text.match(/[ހ-޿]/g) || []).length;
    return (thaana / nonWs) > 0.4;
}

// ── Alpine.js app factory ─────────────────────────────────────────────────────
function workbenchApp() {
    return {
        // Input
        sourceText: '',
        targetLang: 'auto',
        mode: 'faithful',
        model: 'sonnet',
        nCandidates: 1,
        verify: false,
        multiModel: false,
        thaanaKeyboard: false,
        editingSource: false,
        hoveredSourceWord: null,
        hoveredTargetWord: null,
        neuralNetNodes: [],
        neuralNetLinks: [],
        neuralNetLoopId: null,

        // Output
        loading: false,
        result: null,
        error: null,
        nerEntities: [],
        nerLoading: false,
        speaking: false,
        batchAlignment: [],
        batchAlignLoading: false,
        batchAlignError: '',
        wordAlternatives: [],
        altLoading: false,

        // Fluency scoring
        fluencyScore: null,
        fluencyPerplexity: null,
        fluencyLoading: false,

        // Benchmarks
        benchmarks: null,
        benchmarksLoading: false,

        // Word interaction
        activeTab: 'provenance',
        selectedWord: null,
        wordGlossary: [],
        wordConcordance: [],
        wordLoading: false,
        alignedSourceWords: [],
        alignExplanation: '',
        alignLoading: false,

        // Glossary browser
        glossarySearch: '',
        glossaryResults: [],
        glossaryTotal: 0,
        glossaryLoading: false,

        // Spell checker
        spellIssues: [],
        spellLoading: false,
        spellDone: false,

        // History panel
        historyOpen: false,
        historyRuns: [],
        historyLoading: false,

        // Glossary anchor highlighting toggle
        showGlossaryAnchors: false,

        // Token Map — per-card alternatives
        cardAlternatives: {},   // word → [{text, note}]  (present = loaded)
        cardAltLoading: {},     // word → true while fetching

        // ── Computed ────────────────────────────────────────────────────────
        get tokens() {
            return tokenizeDV(this.result?.translation || '');
        },
        get isDV() {
            return this.result?.target_lang === 'DV';
        },
        get entityCheck() {
            return this.result?.entity_check || null;
        },
        get registeredTokens() {
            return this.tokens.filter(t => t.register !== null);
        },
        get charCount() {
            return this.sourceText.length;
        },
        get wordCount() {
            return this.tokens.length;
        },
        get sourceWordCount() {
            return (this.sourceText || '').trim().split(/\s+/).filter(Boolean).length;
        },
        get registerDistribution() {
            const counts = {};
            for (const tok of this.registeredTokens) {
                const key = tok.register.cls;
                if (!counts[key]) counts[key] = { cls: tok.register.cls, label: tok.register.label, count: 0 };
                counts[key].count++;
            }
            return Object.values(counts).sort((a, b) => b.count - a.count);
        },
        get nerTypeSummary() {
            const counts = {};
            for (const e of this.nerEntities) {
                counts[e.type] = (counts[e.type] || 0) + 1;
            }
            return Object.entries(counts)
                .map(([type, count]) => ({ type, count }))
                .sort((a, b) => b.count - a.count);
        },
        get costSparkline() {
            const runs = [...this.historyRuns].reverse().slice(0, 20);
            if (runs.length < 2) return '';
            const costs = runs.map(r => r.cost_usd || 0);
            const max = Math.max(...costs, 0.0001);
            const W = 200, H = 28, PAD = 2;
            const xStep = (W - PAD * 2) / Math.max(costs.length - 1, 1);
            const pts = costs.map((c, i) => {
                const x = (PAD + i * xStep).toFixed(1);
                const y = (H - PAD - (c / max) * (H - PAD * 2)).toFixed(1);
                return `${x},${y}`;
            }).join(' ');
            const dots = costs.map((c, i) => {
                const x = (PAD + i * xStep).toFixed(1);
                const y = (H - PAD - (c / max) * (H - PAD * 2)).toFixed(1);
                return `<circle cx="${x}" cy="${y}" r="2.5" fill="var(--primary)" opacity="0.9"/>`;
            }).join('');
            return `<svg width="${W}" height="${H}" style="display:block;overflow:visible;"><polyline points="${pts}" fill="none" stroke="var(--primary)" stroke-width="1.5" opacity="0.7"/>${dots}</svg>`;
        },
        get sourceTokens() {
            const text = this.sourceText || '';
            return text.split(/\s+/).filter(t => t.length > 0).map(word => {
                const clean = word.replace(/[.,;:!?'"()\[\]]+/g, '').trim();
                return { text: word, clean: clean || word };
            });
        },
        get sourceAlignmentMap() {
            const m = {};
            for (const a of this.batchAlignment) {
                const tgt = (a.target_word || '').toLowerCase();
                for (const sw of (a.source_words || [])) {
                    const swL = sw.toLowerCase();
                    if (!m[swL]) m[swL] = [];
                    m[swL].push(tgt);
                }
            }
            return m;
        },
        hoverSource(word) {
            this.hoveredSourceWord = word.toLowerCase();
            this.hoveredTargetWord = null;
        },
        hoverTarget(word) {
            this.hoveredTargetWord = word.toLowerCase();
            this.hoveredSourceWord = null;
        },
        clearHover() {
            this.hoveredSourceWord = null;
            this.hoveredTargetWord = null;
        },
        isTargetHighlighted(cleanWord) {
            if (!cleanWord) return false;
            const w = cleanWord.toLowerCase();
            if (this.hoveredTargetWord === w) return true;
            if (this.hoveredSourceWord) {
                const alignedTgts = this.sourceAlignmentMap[this.hoveredSourceWord] || [];
                return alignedTgts.includes(w);
            }
            return false;
        },
        isSourceHighlighted(cleanWord) {
            if (!cleanWord) return false;
            const w = cleanWord.toLowerCase();
            if (this.hoveredSourceWord === w) return true;
            if (this.hoveredTargetWord) {
                const alignedSrcs = this.alignmentMap[this.hoveredTargetWord] || [];
                return alignedSrcs.some(sw => sw.toLowerCase() === w);
            }
            return false;
        },
        isAligned(cleanWord) {
            return this.alignedSourceWords.some(
                w => w.toLowerCase() === cleanWord.toLowerCase()
            );
        },
        entityTypeOf(cleanWord) {
            const lw = cleanWord.toLowerCase();
            for (const ent of this.nerEntities) {
                const words = (ent.text || '').toLowerCase().split(/\s+/);
                if (words.includes(lw)) return ent.type;
            }
            return null;
        },

        get alignmentSVG() {
            const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
            const src = this.sourceTokens;
            const tgt = this.tokens;
            if (!src.length || !tgt.length || !this.batchAlignment.length) return '';

            // Wider canvas so Dhivehi labels (right side) have enough room
            const ROW = 36, PAD = 16, W = 760;
            // Left column: EN source labels (text-anchor: end, so text flows LEFT of the dot)
            // Right column: DV target labels (text-anchor: start, left-to-right from dot)
            // Thaana is RTL by Unicode properties; SVG respects Unicode BiDi automatically
            const LABEL_W = 220;               // max label width before arc zone
            const ARC_L = LABEL_W + 12;        // left arc terminus x
            const ARC_R = W - LABEL_W - 12;    // right arc terminus x
            const H = Math.max(src.length, tgt.length) * ROW + PAD * 2;
            const sy = i => PAD + i * ROW + ROW / 2;
            const ty = i => PAD + i * ROW + ROW / 2;

            const ARC_COLORS = ['#C99A4D','#5BA8D5','#5BC787','#9B6BD3','#F08090','#E98050'];
            let colorIdx = 0, arcs = '', srcDots = '', tgtDots = '';
            const isAnyHovered = !!(this.hoveredTargetWord || this.hoveredSourceWord);

            for (const a of this.batchAlignment) {
                const tIdx = tgt.findIndex(t =>
                    t.clean === a.target_word || t.text === a.target_word ||
                    (t.clean && a.target_word.includes(t.clean)) ||
                    (a.target_word && t.text.includes(a.target_word))
                );
                if (tIdx === -1) continue;
                const baseColor = ARC_COLORS[colorIdx++ % ARC_COLORS.length];
                const tyVal = ty(tIdx);
                for (const sw of (a.source_words || [])) {
                    const sIdx = src.findIndex(t =>
                        t.clean.toLowerCase() === sw.toLowerCase() ||
                        t.text.toLowerCase() === sw.toLowerCase()
                    );
                    if (sIdx === -1) continue;
                    const syVal = sy(sIdx);
                    const cx1 = ARC_L + (ARC_R - ARC_L) * 0.3;
                    const cx2 = ARC_L + (ARC_R - ARC_L) * 0.7;

                    // Hover checks
                    const targetMatches = this.hoveredTargetWord && (
                        a.target_word.toLowerCase() === this.hoveredTargetWord ||
                        a.target_word.toLowerCase().includes(this.hoveredTargetWord) ||
                        this.hoveredTargetWord.includes(a.target_word.toLowerCase())
                    );
                    const sourceMatches = this.hoveredSourceWord && sw.toLowerCase() === this.hoveredSourceWord;
                    const isHighlighted = targetMatches || sourceMatches;

                    const color = isHighlighted ? 'var(--primary)' : baseColor;
                    const opacity = isHighlighted ? '0.95' : (isAnyHovered ? '0.08' : '0.65');
                    const strokeWidth = isHighlighted ? '3.5' : '2';

                    arcs += `<path d="M${ARC_L},${syVal} C${cx1},${syVal} ${cx2},${tyVal} ${ARC_R},${tyVal}" stroke="${color}" stroke-width="${strokeWidth}" fill="none" opacity="${opacity}" style="transition: all 0.2s ease;"/>`;
                    srcDots += `<circle cx="${ARC_L}" cy="${syVal}" r="${isHighlighted ? 5 : 4}" fill="${color}" opacity="${opacity}" style="transition: all 0.2s ease;"/>`;
                    tgtDots += `<circle cx="${ARC_R}" cy="${tyVal}" r="${isHighlighted ? 5 : 4}" fill="${color}" opacity="${opacity}" style="transition: all 0.2s ease;"/>`;
                }
            }

            // EN labels: right-aligned, ending at ARC_L - 10
            const srcLabels = src.map((t, i) => {
                const isWordHighlighted = this.hoveredSourceWord === t.clean.toLowerCase() || (
                    this.hoveredTargetWord && (this.alignmentMap[this.hoveredTargetWord] || []).some(sw => sw.toLowerCase() === t.clean.toLowerCase())
                );
                const opacity = isWordHighlighted ? '1' : (isAnyHovered ? '0.3' : '0.85');
                const fontWeight = isWordHighlighted ? '600' : '400';
                const color = isWordHighlighted ? 'var(--fg)' : 'var(--fg-dim)';
                return `<text x="${ARC_L - 10}" y="${sy(i) + 5}" text-anchor="end" class="arc-src" fill="${color}" opacity="${opacity}" font-weight="${fontWeight}" style="cursor:pointer; transition: all 0.15s ease;" onmouseenter="window.workbenchAppInstance?.hoverSource('${t.clean.replace(/'/g, "\\'")}')" onmouseleave="window.workbenchAppInstance?.clearHover()">${esc(t.text)}</text>`;
            }).join('');

            // DV labels: left-aligned starting at ARC_R + 10
            const tgtLabels = tgt.map((t, i) => {
                const isWordHighlighted = this.hoveredTargetWord === t.clean.toLowerCase() || (
                    this.hoveredSourceWord && (this.sourceAlignmentMap[this.hoveredSourceWord] || []).includes(t.clean.toLowerCase())
                );
                const opacity = isWordHighlighted ? '1' : (isAnyHovered ? '0.3' : '0.85');
                const fontWeight = isWordHighlighted ? '600' : '400';
                const color = isWordHighlighted ? 'var(--fg)' : 'var(--fg-dim)';
                return `<text x="${ARC_R + 10}" y="${ty(i) + 5}" text-anchor="start" class="arc-tgt" unicode-bidi="plaintext" fill="${color}" opacity="${opacity}" font-weight="${fontWeight}" style="cursor:pointer; transition: all 0.15s ease;" onmouseenter="window.workbenchAppInstance?.hoverTarget('${t.clean.replace(/'/g, "\\'")}')" onmouseleave="window.workbenchAppInstance?.clearHover()">${esc(t.text)}</text>`;
            }).join('');

            // viewBox makes max-width:100% actually scale the SVG on narrow panels
            return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" class="alignment-arc-svg">${arcs}${srcDots}${tgtDots}${srcLabels}${tgtLabels}</svg>`;
        },

        isSpellingIssue(cleanWord) {
            return this.spellIssues.some(
                s => s.word === cleanWord || s.word.replace(/[،.؟!]+/g, '') === cleanWord
            );
        },
        spellSuggestionFor(cleanWord) {
            const issue = this.spellIssues.find(
                s => s.word === cleanWord || s.word.replace(/[،.؟!]+/g, '') === cleanWord
            );
            return issue ? `${issue.suggestion} — ${issue.reason}` : '';
        },

        // ── Translate ────────────────────────────────────────────────────────
        async doTranslate() {
            if (!this.sourceText.trim()) return;
            if (this.neuralNetLoopId) {
                cancelAnimationFrame(this.neuralNetLoopId);
                this.neuralNetLoopId = null;
            }
            this.editingSource = false;
            this.loading = true;
            this.result = null;
            this.error = null;
            this.selectedWord = null;
            this.wordGlossary = [];
            this.wordConcordance = [];
            this.wordAlternatives = [];
            this.spellIssues = [];
            this.spellDone = false;
            this.cardAlternatives = {};
            this.cardAltLoading = {};
            this.fluencyScore = null;
            this.fluencyPerplexity = null;
            this.fluencyLoading = false;
            this.activeTab = 'provenance';
            try {
                const r = await fetch('/api/translate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: this.sourceText,
                        target_language: this.targetLang,
                        verify: this.verify,
                        mode: this.mode,
                        model: this.model,
                        n_candidates: +this.nCandidates || 1,
                        multi_model: this.multiModel,
                    }),
                });
                const data = await r.json();
                if (!r.ok) {
                    this.error = data.detail || data.error || `Error ${r.status}`;
                    return;
                }
                this.result = data;
                // Auto-run NER + batch alignment + fluency (non-blocking, parallel)
                this._runNER(data.translation, data.target_lang || 'DV');
                this._runBatchAlign(this.sourceText, data.translation, data.source_lang || 'EN', data.target_lang || 'DV');
                this._runFluency(data.translation, data.target_lang || 'DV');
                // Scroll analysis panel into view
                this.$nextTick(() => {
                    const panel = this.$el.querySelector('.wb-analysis');
                    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
                });
            } catch (e) {
                this.error = e.message || String(e);
            } finally {
                this.loading = false;
            }
        },

        async _runBatchAlign(source, translation, srcLang, tgtLang) {
            if (!source || !translation) return;
            this.batchAlignment = [];
            this.batchAlignError = '';
            this.batchAlignLoading = true;
            try {
                const r = await fetch('/api/align-batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source, translation, source_lang: srcLang, target_lang: tgtLang }),
                });
                if (!r.ok) {
                    // Rate-limit handler returns plain text; other errors return JSON
                    const raw = await r.text();
                    let msg;
                    try { msg = JSON.parse(raw).detail; } catch { msg = raw; }
                    this.batchAlignError = msg || `Alignment error ${r.status}`;
                    return;
                }
                const data = await r.json();
                this.batchAlignment = data.alignments || [];
            } catch (e) {
                this.batchAlignError = 'Alignment unavailable — check server connectivity';
            } finally {
                this.batchAlignLoading = false;
            }
        },

        async _runNER(text, lang) {
            if (!text) return;
            this.nerEntities = [];
            this.nerLoading = true;
            try {
                const data = await fetch('/api/ner', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, lang }),
                }).then(r => r.json());
                this.nerEntities = data.entities || [];
            } catch (_) {
                // non-fatal
            } finally {
                this.nerLoading = false;
            }
        },

        async _runFluency(text, lang) {
            if (!text || lang !== 'DV') return;
            this.fluencyLoading = true;
            try {
                const data = await fetch('/api/fluency', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text }),
                }).then(r => r.json());
                if (data.available) {
                    this.fluencyScore = data.fluency_score;
                    this.fluencyPerplexity = data.perplexity;
                }
            } catch (_) {
                // non-fatal — fluency panel stays hidden
            } finally {
                this.fluencyLoading = false;
            }
        },

        speakTranslation() {
            if (!this.result?.translation || !window.speechSynthesis) return;
            if (this.speaking) {
                speechSynthesis.cancel();
                this.speaking = false;
                return;
            }
            const utt = new SpeechSynthesisUtterance(this.result.translation);
            utt.lang = this.result.target_lang === 'DV' ? 'dv' : 'en-US';
            utt.onend = () => { this.speaking = false; };
            utt.onerror = () => { this.speaking = false; };
            this.speaking = true;
            speechSynthesis.speak(utt);
        },

        // ── Word click ───────────────────────────────────────────────────────
        async selectWord(word) {
            const clean = word.replace(/[.،؟!،؟۔"'()\[\]]+/g, '').trim();
            if (!clean) return;
            this.selectedWord = clean;
            this.activeTab = 'word-detail';
            this.wordLoading = true;
            this.wordGlossary = [];
            this.wordConcordance = [];
            this.wordAlternatives = [];
            this.alignedSourceWords = [];
            this.alignExplanation = '';
            this.alignLoading = true;
            this.altLoading = true;
            const lang = this.result?.target_lang || 'DV';
            try {
                const [concData, glossData, altData] = await Promise.all([
                    fetch(`/api/concordance?q=${encodeURIComponent(clean)}&lang=${lang}&limit=5`)
                        .then(r => r.json()),
                    fetch(`/api/glossary?q=${encodeURIComponent(clean)}&limit=5`)
                        .then(r => r.json()),
                    fetch('/api/alternatives', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            word: clean,
                            translation: this.result?.translation || '',
                            source: this.sourceText || '',
                            target_lang: lang,
                        }),
                    }).then(r => r.json()),
                ]);
                this.wordConcordance = concData.results || [];
                this.wordGlossary = glossData.terms || [];
                this.wordAlternatives = altData.alternatives || [];
            } catch (_) {
                // non-fatal — panels stay empty
            } finally {
                this.wordLoading = false;
                this.altLoading = false;
            }
            // Word-level alignment is derived from batch alignments.
            // If batch alignments are not available yet, fetch them once.
            if (this.sourceText && this.result?.translation) {
                try {
                    if (!this.batchAlignment.length) {
                        const r = await fetch('/api/align-batch', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                source: this.sourceText,
                                translation: this.result.translation,
                                source_lang: this.result.source_lang || 'EN',
                                target_lang: this.result.target_lang || 'DV',
                            }),
                        });
                        if (r.ok) {
                            const data = await r.json();
                            this.batchAlignment = data.alignments || [];
                        }
                    }

                    const needle = clean.toLowerCase();
                    const srcWords = new Set();
                    for (const a of this.batchAlignment) {
                        const target = (a.target_word || '').replace(/[.،؟!،؟۔"'()\[\]]+/g, '').trim().toLowerCase();
                        if (!target) continue;
                        if (target === needle || target.includes(needle) || needle.includes(target)) {
                            for (const sw of (a.source_words || [])) {
                                if (sw) srcWords.add(sw);
                            }
                        }
                    }
                    this.alignedSourceWords = [...srcWords];
                    this.alignExplanation = this.alignedSourceWords.length
                        ? 'Derived from batch alignment for this translation.'
                        : '';
                } catch (_) {
                    // non-fatal
                } finally {
                    this.alignLoading = false;
                }
            } else {
                this.alignLoading = false;
            }
        },

        // ── Token Map card click ─────────────────────────────────────────────
        async selectCard(tok) {
            const word = tok.clean;
            if (!word || this.cardAltLoading[word]) return;
            // Toggle off if already loaded
            if (this.hasLoadedCard(word)) {
                const next = { ...this.cardAlternatives };
                delete next[word];
                this.cardAlternatives = next;
                return;
            }
            this.cardAltLoading = { ...this.cardAltLoading, [word]: true };
            try {
                const data = await fetch('/api/alternatives', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        word,
                        translation: this.result?.translation || '',
                        source: this.sourceText || '',
                        target_lang: this.result?.target_lang || 'DV',
                    }),
                }).then(r => r.json());
                this.cardAlternatives = { ...this.cardAlternatives, [word]: data.alternatives || [] };
            } catch (_) {
                this.cardAlternatives = { ...this.cardAlternatives, [word]: [] };
            } finally {
                const next = { ...this.cardAltLoading };
                delete next[word];
                this.cardAltLoading = next;
            }
        },

        // ── Glossary browser ─────────────────────────────────────────────────
        async searchGlossary() {
            this.glossaryLoading = true;
            try {
                const q = this.glossarySearch.trim();
                const url = q
                    ? `/api/glossary?q=${encodeURIComponent(q)}&limit=50`
                    : '/api/glossary?limit=50';
                const data = await fetch(url).then(r => r.json());
                this.glossaryResults = data.terms || [];
                this.glossaryTotal = data.total || 0;
            } catch (_) {
                this.glossaryResults = [];
            } finally {
                this.glossaryLoading = false;
            }
        },

        // ── Utilities ────────────────────────────────────────────────────────
        copyTranslation() {
            if (this.result?.translation) {
                navigator.clipboard.writeText(this.result.translation);
            }
        },

        downloadTranslation() {
            if (!this.result?.translation) return;
            const blob = new Blob([this.result.translation], { type: 'text/plain;charset=utf-8' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `translation_${this.result.target_lang}_${Date.now()}.txt`;
            a.click();
            URL.revokeObjectURL(a.href);
        },

        clearAll() {
            if (this.neuralNetLoopId) {
                cancelAnimationFrame(this.neuralNetLoopId);
                this.neuralNetLoopId = null;
            }
            this.editingSource = false;
            this.sourceText = '';
            this.result = null;
            this.error = null;
            this.nerEntities = [];
            this.batchAlignment = [];
            this.batchAlignError = '';
            this.spellIssues = [];
            this.spellDone = false;
            this.selectedWord = null;
            this.wordGlossary = [];
            this.wordConcordance = [];
            this.wordAlternatives = [];
            this.alignedSourceWords = [];
            this.alignExplanation = '';
            this.wordLoading = false;
            this.alignLoading = false;
            this.altLoading = false;
            this.nerLoading = false;
            this.batchAlignLoading = false;
            this.cardAlternatives = {};
            this.cardAltLoading = {};
            this.fluencyScore = null;
            this.fluencyPerplexity = null;
            this.fluencyLoading = false;
            this.activeTab = 'provenance';
        },

        toggleThaanaKeyboard() {
            this.thaanaKeyboard = !this.thaanaKeyboard;
            const ta = this.$el.querySelector('textarea.wb-input');
            if (ta) ThaanaKeyboard.toggle(ta, this.thaanaKeyboard);
        },

        // Map target_word → source_words[] from batchAlignment (lowercased keys)
        get alignmentMap() {
            const m = {};
            for (const a of this.batchAlignment) {
                m[(a.target_word || '').toLowerCase()] = a.source_words || [];
            }
            return m;
        },
        // Find source words for a given DV token, with fuzzy fallback
        sourceWordsFor(cleanWord) {
            if (!cleanWord) return [];
            const lw = cleanWord.toLowerCase();
            if (this.alignmentMap[lw]) return this.alignmentMap[lw];
            // Fuzzy: alignment key may include punctuation the tokenizer stripped
            for (const [key, words] of Object.entries(this.alignmentMap)) {
                if (key.length > 1 && (key.includes(lw) || lw.includes(key))) return words;
            }
            return [];
        },
        hasLoadedCard(word) {
            return Object.prototype.hasOwnProperty.call(this.cardAlternatives, word);
        },

        get glossaryAnchoredSet() {
            const s = new Set();
            for (const gt of (this.result?.glossary_terms || [])) {
                if (gt.dv_term) s.add(gt.dv_term.trim().toLowerCase());
            }
            return s;
        },
        isGlossaryAnchored(cleanWord) {
            return this.glossaryAnchoredSet.has((cleanWord || '').toLowerCase());
        },

        get backTranslationDiff() {
            const src = (this.sourceText || '').trim();
            const bt  = (this.result?.verification?.back_translation || '').trim();
            if (!src || !bt) return [];
            const wa = src.split(/\s+/), wb = bt.split(/\s+/);
            const n = wa.length, m = wb.length;
            // LCS table
            const dp = Array.from({length: n + 1}, () => new Int16Array(m + 1));
            for (let i = 1; i <= n; i++)
                for (let j = 1; j <= m; j++)
                    dp[i][j] = wa[i-1].toLowerCase() === wb[j-1].toLowerCase()
                        ? dp[i-1][j-1] + 1
                        : Math.max(dp[i-1][j], dp[i][j-1]);
            // Traceback
            const out = [];
            let i = n, j = m;
            while (i > 0 || j > 0) {
                if (i > 0 && j > 0 && wa[i-1].toLowerCase() === wb[j-1].toLowerCase()) {
                    out.unshift({ text: wa[i-1], type: 'same' });
                    i--; j--;
                } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
                    out.unshift({ text: wb[j-1], type: 'ins' });
                    j--;
                } else {
                    out.unshift({ text: wa[i-1], type: 'del' });
                    i--;
                }
            }
            return out;
        },

        regBarColor(cls) {
            const map = {
                'badge-purple': 'rgba(155,107,211,0.65)',
                'badge-blue':   'rgba(91,168,213,0.65)',
                'badge-yellow': 'rgba(247,201,72,0.65)',
                'badge-amber':  'rgba(240,160,75,0.65)',
                'badge-dim':    'rgba(107,130,154,0.45)',
            };
            return map[cls] || 'rgba(107,130,154,0.3)';
        },

        fmtCost(v) {
            return v != null ? '$' + Number(v).toFixed(4) : '';
        },

        fmtConf(v) {
            return v != null ? Math.round(v * 100) + '%' : '—';
        },

        // ── Spell checker ────────────────────────────────────────────────────
        async runSpellCheck() {
            if (!this.result?.translation) return;
            this.spellLoading = true;
            this.spellIssues = [];
            this.spellDone = false;
            try {
                const data = await fetch('/api/spellcheck', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: this.result.translation,
                        lang: this.result.target_lang || 'DV',
                    }),
                }).then(r => r.json());
                this.spellIssues = data.issues || [];
                this.spellDone = true;
            } catch (_) {
                this.spellDone = true;
            } finally {
                this.spellLoading = false;
            }
        },

        // ── History panel ────────────────────────────────────────────────────
        async loadHistory() {
            this.historyLoading = true;
            this.historyRuns = [];
            try {
                const data = await fetch('/api/translate/history?limit=20').then(r => r.json());
                this.historyRuns = data.runs || [];
            } catch (_) {
            } finally {
                this.historyLoading = false;
            }
        },

        loadHistoryRun(run) {
            this.sourceText = run.source_text || '';
            this.result = {
                translation: run.translation,
                source_lang: run.source_lang || '',
                target_lang: run.target_lang,
                model: run.model,
                cost_usd: run.cost_usd,
            };
            this.historyOpen = false;
            this.activeTab = 'provenance';
            this.nerEntities = [];
            this.batchAlignment = [];
            this.spellIssues = [];
            this.spellDone = false;
        },

        toggleHistory() {
            this.historyOpen = !this.historyOpen;
            if (this.historyOpen && !this.historyRuns.length) this.loadHistory();
        },

        async loadBenchmarks() {
            if (this.benchmarksLoading) return;
            this.benchmarksLoading = true;
            try {
                const data = await fetch('/api/benchmarks').then(r => r.json());
                this.benchmarks = data;
            } catch (_) {
                // non-fatal
            } finally {
                this.benchmarksLoading = false;
            }
        },

        // ── Export ───────────────────────────────────────────────────────────
        exportJSON() {
            if (!this.result) return;
            const bundle = {
                source: this.sourceText,
                translation: this.result.translation,
                target_lang: this.result.target_lang,
                source_lang: this.result.source_lang,
                model: this.result.model,
                mode: this.result.mode,
                cost_usd: this.result.cost_usd,
                tokens_in: this.result.tokens_in,
                tokens_out: this.result.tokens_out,
                n_candidates: this.result.n_candidates,
                terms_locked: this.result.terms_locked,
                lock_misses: this.result.lock_misses,
                fluency_score: this.fluencyScore,
                fluency_perplexity: this.fluencyPerplexity,
                ner_entities: this.nerEntities,
                spell_issues: this.spellIssues,
                alignment: this.batchAlignment,
                glossary_terms: this.result.glossary_terms || [],
                exported_at: new Date().toISOString(),
            };
            const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `translation_bundle_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(a.href);
        },

        // ── Benchmarks computed ──────────────────────────────────────────────
        get benchmarkMainRows() {
            if (!this.benchmarks) return [];
            const rows = [];
            for (const [runId, run] of Object.entries(this.benchmarks.runs || {})) {
                const agg = run.data?.main_set_aggregate || {};
                for (const [sys, stats] of Object.entries(agg)) {
                    if (sys.startsWith('_')) continue;
                    rows.push({
                        runId,
                        run_label: run.label,
                        sys: sys.replace(/_raw$/, '').replace(/_/g, ' '),
                        isMoonlight: runId === 'moonlight_full',
                        chrf: stats.chrf?.mean != null ? stats.chrf.mean.toFixed(1) : '—',
                        bleu: stats.bleu?.mean != null ? stats.bleu.mean.toFixed(1) : '—',
                        fluency: (
                            stats.fluency_diagnostic?.mean ?? stats.fluency_mean
                        ) != null
                            ? (stats.fluency_diagnostic?.mean ?? stats.fluency_mean).toFixed(0)
                            : '—',
                        cost: stats.total_cost_usd != null ? '$' + stats.total_cost_usd.toFixed(2) : '',
                    });
                }
            }
            return rows;
        },
        get benchmarkChallengeRows() {
            if (!this.benchmarks) return [];
            const cats = this.benchmarks.runs?.moonlight_full?.data
                ?.challenge_set_aggregate?.moonlight_full?.by_category || {};
            return Object.entries(cats)
                .map(([cat, acc]) => ({
                    label: cat.replace(/^cat\d+_/, '').replace(/_/g, ' '),
                    acc,
                    pct: Math.round(acc * 100),
                    color: acc >= 0.8 ? 'var(--good)' : acc >= 0.6 ? 'var(--info)' : 'var(--deadline)',
                }))
                .sort((a, b) => b.acc - a.acc);
        },
        get benchmarkChallengeOverall() {
            if (!this.benchmarks) return null;
            return this.benchmarks.runs?.moonlight_full?.data
                ?.challenge_set_aggregate?.moonlight_full || null;
        },

        initNeuralNet() {
            const canvas = document.getElementById('neuralNetCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            if (this.neuralNetLoopId) {
                cancelAnimationFrame(this.neuralNetLoopId);
                this.neuralNetLoopId = null;
            }

            const src = this.sourceTokens;
            const tgt = this.tokens;
            if (!src.length || !tgt.length) return;

            // Fit canvas to container width
            const W = Math.max(560, (canvas.parentElement?.clientWidth || 760));
            const H = Math.round(W * 0.56);
            canvas.width = W;
            canvas.height = H;

            // Reuse existing layout when translation + alignment data hasn't changed
            const cacheKey = (this.result?.translation || '') + '|' + this.sourceText + '|' + this.batchAlignment.length;
            let settled = false;

            if (this._nnCacheKey === cacheKey && this.neuralNetNodes.length > 0) {
                settled = true; // already laid out — skip rebuild
            } else {
                this._nnCacheKey = cacheKey;
                const nodes = [], links = [];
                const totalWords = src.length + tgt.length;
                const R = Math.max(14, Math.min(26, Math.floor(260 / Math.max(totalWords, 10))));

                src.forEach((t, i) => {
                    const ty = H * 0.1 + (i / Math.max(src.length - 1, 1)) * H * 0.8;
                    nodes.push({
                        id: `src_${i}`, label: t.text, clean: t.clean, type: 'source',
                        x: W * 0.2 + (Math.random() - 0.5) * 8, y: ty,
                        targetX: W * 0.2, targetY: ty, vx: 0, vy: 0, radius: R,
                    });
                });

                tgt.forEach((t, i) => {
                    const ty = H * 0.1 + (i / Math.max(tgt.length - 1, 1)) * H * 0.8;
                    nodes.push({
                        id: `tgt_${i}`, label: t.text, clean: t.clean,
                        register: t.register, type: 'target',
                        x: W * 0.8 + (Math.random() - 0.5) * 8, y: ty,
                        targetX: W * 0.8, targetY: ty, vx: 0, vy: 0, radius: R + 2,
                    });
                });

                // Alignment links — exact clean-match first, partial only for longer tokens
                for (const a of this.batchAlignment) {
                    const needle = (a.target_word || '').toLowerCase();
                    const tNode = nodes.find(n => {
                        if (n.type !== 'target') return false;
                        const nc = n.clean.toLowerCase(), nl = n.label.toLowerCase();
                        if (nc === needle || nl === needle) return true;
                        if (needle.length > 4) return nc.includes(needle) || needle.includes(nc);
                        return false;
                    });
                    if (!tNode) continue;
                    for (const sw of (a.source_words || [])) {
                        const swL = sw.toLowerCase();
                        const sNode = nodes.find(n => {
                            if (n.type !== 'source') return false;
                            const nc = n.clean.toLowerCase();
                            return nc === swL || (swL.length > 3 && nc.includes(swL));
                        });
                        if (sNode) links.push({ source: sNode, target: tNode, type: 'alignment', weight: 1.0 });
                    }
                }

                // NER entity nodes (DV entities from translation) — upper middle column
                const entities = this.nerEntities || [];
                entities.forEach((ent, i) => {
                    const ty = H * 0.1 + (i / Math.max(entities.length, 1)) * H * 0.38;
                    const entNode = {
                        id: `ner_${i}`, label: ent.type, subtitle: ent.text || '',
                        fullLabel: `[${ent.type}] ${ent.text || ''}`,
                        type: 'entity', x: W * 0.5, y: ty, targetX: W * 0.5, targetY: ty,
                        vx: 0, vy: 0, radius: R,
                    };
                    nodes.push(entNode);
                    const entWords = (ent.text || '').toLowerCase().split(/\s+/);
                    tgt.forEach(t => {
                        const tc = t.clean.toLowerCase();
                        if (entWords.some(w => w.length > 2 && (tc === w || tc.includes(w) || w.includes(tc)))) {
                            const tn = nodes.find(n => n.type === 'target' && n.clean === t.clean);
                            if (tn) links.push({ source: entNode, target: tn, type: 'entity-link', weight: 0.6 });
                        }
                    });
                });

                // Register nodes — lower middle column
                const uniqueRegs = {};
                tgt.forEach(t => { if (t.register) uniqueRegs[t.register.label] = t.register.cls; });
                const regEntries = Object.entries(uniqueRegs);
                regEntries.forEach(([label, cls], i) => {
                    const ty = H * 0.55 + (i / Math.max(regEntries.length, 1)) * H * 0.32;
                    const regNode = {
                        id: `reg_${label}`, label: label.toUpperCase(),
                        type: 'register', cls, x: W * 0.5, y: ty,
                        targetX: W * 0.5, targetY: ty, vx: 0, vy: 0, radius: R,
                    };
                    nodes.push(regNode);
                    tgt.forEach(t => {
                        if (t.register?.label === label) {
                            const tn = nodes.find(n => n.type === 'target' && n.clean === t.clean);
                            if (tn) links.push({ source: regNode, target: tn, type: 'register-link', weight: 0.5 });
                        }
                    });
                });

                this.neuralNetNodes = nodes;
                this.neuralNetLinks = links;
            }

            // ── Interaction ───────────────────────────────────────────────────
            let draggedNode = null, hoveredNode = null;
            const getPos = e => {
                const r = canvas.getBoundingClientRect();
                return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) };
            };
            canvas.onmousedown = e => {
                const p = getPos(e);
                let best = null, bd = 42;
                this.neuralNetNodes.forEach(n => { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n; } });
                draggedNode = best;
                if (draggedNode) { draggedNode.vx = 0; draggedNode.vy = 0; settled = false; }
            };
            canvas.onmousemove = e => {
                const p = getPos(e);
                if (draggedNode) { draggedNode.x = p.x; draggedNode.y = p.y; return; }
                let best = null, bd = 36;
                this.neuralNetNodes.forEach(n => { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n; } });
                if (hoveredNode !== best) {
                    hoveredNode = best;
                    if (hoveredNode?.type === 'target') this.hoverTarget(hoveredNode.label);
                    else if (hoveredNode?.type === 'source') this.hoverSource(hoveredNode.label);
                    else this.clearHover();
                }
            };
            canvas.onmouseup = () => { draggedNode = null; };
            canvas.onmouseleave = () => { draggedNode = null; hoveredNode = null; this.clearHover(); };

            // ── Physics constants ─────────────────────────────────────────────
            const friction = 0.80, repulsionK = 1800, springLen = 90, springK = 0.035;

            // ── Draw helpers ──────────────────────────────────────────────────
            const dvFont = '"Noto Sans Thaana","Faruma","MV Boli",sans-serif';
            const enFont = '"Plus Jakarta Sans","Outfit",sans-serif';

            // Bezier S-curve for alignment links, with optional arrowhead at target end
            const drawBezier = (sx, sy, tx, ty, color, width, arrow) => {
                const tension = Math.abs(tx - sx) * 0.42;
                ctx.strokeStyle = color; ctx.lineWidth = width;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.bezierCurveTo(sx + tension, sy, tx - tension, ty, tx, ty);
                ctx.stroke();
                if (arrow) {
                    // End tangent direction is +x (bezier exits cp2 horizontally)
                    const al = 7, as = 0.38;
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.moveTo(tx, ty);
                    ctx.lineTo(tx - al * Math.cos(-as), ty - al * Math.sin(-as));
                    ctx.lineTo(tx - al * Math.cos(as),  ty - al * Math.sin(as));
                    ctx.closePath();
                    ctx.fill();
                }
            };

            const LEGEND = [
                { color: 'rgba(108,142,247,0.7)', label: 'Source (EN)' },
                { color: 'rgba(56,178,172,0.7)',  label: 'Target (DV)' },
                { color: 'rgba(236,201,75,0.6)',  label: 'Named entity' },
                { color: 'rgba(159,122,234,0.6)', label: 'Register' },
            ];

            const NODE_STYLE = {
                source:   { stroke: 'rgba(108,142,247,0.45)', fill: 'rgba(12,21,36,0.92)',    glow: 'rgba(108,142,247,0.75)' },
                target:   { stroke: 'rgba(56,178,172,0.45)',  fill: 'rgba(12,21,36,0.92)',    glow: 'rgba(56,178,172,0.75)'  },
                entity:   { stroke: 'rgba(236,201,75,0.45)',  fill: 'rgba(236,201,75,0.07)',  glow: 'rgba(236,201,75,0.75)'  },
                register: { stroke: 'rgba(159,122,234,0.45)', fill: 'rgba(159,122,234,0.07)', glow: 'rgba(159,122,234,0.75)' },
            };

            // ── Render loop ───────────────────────────────────────────────────
            const tick = () => {
                if (this.activeTab !== 'neural-net') return; // stop when tab hidden

                const nodes = this.neuralNetNodes;
                const links = this.neuralNetLinks;

                // Physics — skip once settled; drag reactivates
                if (!settled || draggedNode) {
                    let energy = 0;

                    nodes.forEach(n => {
                        if (n === draggedNode) return;
                        const gx = (n.type === 'source' || n.type === 'target') ? 0.06 : 0.025;
                        n.vx += (n.targetX - n.x) * gx;
                        n.vy += (n.targetY - n.y) * 0.03;
                    });

                    for (let i = 0; i < nodes.length; i++) {
                        for (let j = i + 1; j < nodes.length; j++) {
                            const a = nodes[i], b = nodes[j];
                            const dx = b.x - a.x, dy = b.y - a.y;
                            const d = Math.hypot(dx, dy) || 1;
                            if (d < 180) {
                                const f = repulsionK / (d * d);
                                if (a !== draggedNode) { a.vx -= dx/d*f; a.vy -= dy/d*f; }
                                if (b !== draggedNode) { b.vx += dx/d*f; b.vy += dy/d*f; }
                            }
                        }
                    }

                    links.forEach(l => {
                        const dx = l.target.x - l.source.x, dy = l.target.y - l.source.y;
                        const d = Math.hypot(dx, dy) || 1;
                        const f = (d - springLen) * springK * l.weight;
                        const fx = dx/d*f, fy = dy/d*f;
                        if (l.source !== draggedNode) { l.source.vx += fx; l.source.vy += fy; }
                        if (l.target !== draggedNode) { l.target.vx -= fx; l.target.vy -= fy; }
                    });

                    nodes.forEach(n => {
                        if (n === draggedNode) return;
                        n.vx *= friction; n.vy *= friction;
                        n.x += n.vx; n.y += n.vy;
                        const pad = n.radius + 10;
                        n.x = Math.max(pad, Math.min(W - pad, n.x));
                        n.y = Math.max(pad, Math.min(H - pad, n.y));
                        energy += n.vx * n.vx + n.vy * n.vy;
                    });

                    if (energy < 0.1 && !draggedNode) settled = true;
                }
                if (draggedNode) settled = false;

                // ─ Draw ───────────────────────────────────────────────────────
                ctx.clearRect(0, 0, W, H);

                // Subtle grid
                ctx.strokeStyle = 'rgba(255,255,255,0.02)'; ctx.lineWidth = 1;
                for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
                for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

                // Lane dividers
                ctx.save(); ctx.setLineDash([5, 7]); ctx.strokeStyle = 'rgba(255,255,255,0.04)';
                [W * 0.35, W * 0.65].forEach(x => { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); });
                ctx.restore();

                // Column headers
                ctx.save();
                ctx.font = `bold 9px ${enFont}`; ctx.fillStyle = 'rgba(255,255,255,0.2)'; ctx.textAlign = 'center';
                ctx.fillText('SOURCE (EN)', W * 0.2, 16);
                ctx.fillText('FEATURES',    W * 0.5, 16);
                ctx.fillText('TARGET (DV)', W * 0.8, 16);
                ctx.restore();

                // Links
                links.forEach(l => {
                    const linked = hoveredNode && (l.source === hoveredNode || l.target === hoveredNode);
                    const any = !!hoveredNode;
                    let color, width;
                    if (any) {
                        color = linked
                            ? (l.type === 'alignment'   ? 'rgba(108,142,247,0.88)'
                             : l.type === 'entity-link' ? 'rgba(56,178,172,0.88)'
                             :                            'rgba(159,122,234,0.88)')
                            : 'rgba(255,255,255,0.02)';
                        width = linked ? 2.0 : 0.5;
                    } else {
                        color = l.type === 'alignment'   ? 'rgba(108,142,247,0.22)'
                              : l.type === 'entity-link' ? 'rgba(56,178,172,0.20)'
                              :                            'rgba(159,122,234,0.20)';
                        width = 1.0;
                    }
                    if (l.type === 'alignment') {
                        drawBezier(l.source.x, l.source.y, l.target.x, l.target.y, color, width, linked);
                    } else {
                        ctx.strokeStyle = color; ctx.lineWidth = width;
                        ctx.beginPath(); ctx.moveTo(l.source.x, l.source.y); ctx.lineTo(l.target.x, l.target.y); ctx.stroke();
                    }
                });

                // Nodes
                nodes.forEach(n => {
                    const isHov = hoveredNode === n;
                    const isLinked = hoveredNode && links.some(l =>
                        (l.source === hoveredNode && l.target === n) ||
                        (l.target === hoveredNode && l.source === n));
                    const dimmed = !!hoveredNode && !isHov && !isLinked;
                    const s = NODE_STYLE[n.type] || NODE_STYLE.source;

                    ctx.save();
                    ctx.globalAlpha = dimmed ? 0.15 : 1.0;
                    if (isHov) { ctx.shadowBlur = 14; ctx.shadowColor = s.glow; }

                    ctx.beginPath(); ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                    ctx.fillStyle = s.fill; ctx.fill();
                    ctx.lineWidth = isHov ? 2.5 : 1.5;
                    ctx.strokeStyle = isHov ? s.glow : s.stroke; ctx.stroke();
                    ctx.shadowBlur = 0;

                    // Label — auto-shrink to fit inside circle
                    const isTarget = n.type === 'target';
                    let fsize = isTarget ? 13 : 11;
                    const fnt = isTarget ? dvFont : enFont;
                    ctx.font = `bold ${fsize}px ${fnt}`;
                    ctx.fillStyle = 'rgba(255,255,255,0.92)';
                    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                    let lbl = n.label;
                    const maxChars = Math.ceil(n.radius * 1.5);
                    if (lbl.length > maxChars) lbl = lbl.slice(0, maxChars - 1) + '…';
                    const tw = ctx.measureText(lbl).width;
                    const maxW = n.radius * 1.82;
                    if (tw > maxW) {
                        fsize = Math.max(7, Math.floor(fsize * maxW / tw));
                        ctx.font = `bold ${fsize}px ${fnt}`;
                    }
                    ctx.fillText(lbl, n.x, n.y);

                    // Entity subtitle drawn BELOW the circle (not overlapping label text)
                    if (n.type === 'entity' && n.subtitle) {
                        ctx.font = `9px ${enFont}`;
                        ctx.fillStyle = 'rgba(255,255,255,0.38)';
                        let sub = n.subtitle.length > 12 ? n.subtitle.slice(0, 11) + '…' : n.subtitle;
                        ctx.fillText(sub, n.x, n.y + n.radius + 11);
                    }

                    ctx.restore();
                });

                // Tooltip — full label above (or below) hovered node
                if (hoveredNode) {
                    const full = hoveredNode.fullLabel || hoveredNode.label;
                    const isTarget = hoveredNode.type === 'target';
                    ctx.font = `11px ${isTarget ? dvFont : enFont}`;
                    const tw = ctx.measureText(full).width;
                    const pad = 7, th = 20;
                    let tx = Math.max(4, Math.min(W - tw - pad * 2 - 4, hoveredNode.x - tw / 2 - pad));
                    let ty = hoveredNode.y - hoveredNode.radius - th - 8;
                    if (ty < 4) ty = hoveredNode.y + hoveredNode.radius + 8;
                    ctx.fillStyle = 'rgba(0,0,0,0.82)';
                    if (ctx.roundRect) {
                        ctx.beginPath(); ctx.roundRect(tx, ty, tw + pad * 2, th, 4); ctx.fill();
                    } else {
                        ctx.fillRect(tx, ty, tw + pad * 2, th);
                    }
                    ctx.fillStyle = 'rgba(228,232,242,0.95)';
                    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                    ctx.fillText(full, tx + pad, ty + th / 2);
                }

                // Legend — bottom-left corner
                ctx.save();
                const lx = 10, ly = H - LEGEND.length * 17 - 6;
                LEGEND.forEach(({ color, label }, i) => {
                    const y = ly + i * 17;
                    ctx.beginPath(); ctx.arc(lx + 5, y + 6, 5, 0, Math.PI * 2);
                    ctx.fillStyle = color; ctx.fill();
                    ctx.font = `9px ${enFont}`; ctx.fillStyle = 'rgba(255,255,255,0.35)';
                    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                    ctx.fillText(label, lx + 14, y + 6);
                });
                ctx.restore();

                this.neuralNetLoopId = requestAnimationFrame(tick);
            };

            this.neuralNetLoopId = requestAnimationFrame(tick);
        },

        // init() is called automatically by Alpine on component mount
        init() {
            window.workbenchAppInstance = this;
            this.searchGlossary();
            // Pick up source text forwarded from /translate via "Open in Workbench"
            const prefill = sessionStorage.getItem('wb_prefill');
            if (prefill) {
                this.sourceText = prefill;
                sessionStorage.removeItem('wb_prefill');
            }
        },
    };
}
