// SPDX-License-Identifier: MIT
// Thaana phonetic keyboard — adapted from jawish/jtk v4.2.2
// https://github.com/jawish/jtk  (MIT licence, © 2014 Jawish Hameed)
//
// Maps standard QWERTY keys to Unicode Thaana (phonetic layout).
// Key examples:  h→ހ  s→ސ  S→ށ  d→ދ  D→ޑ  a→ަ  A→ާ  i→ި  I→ީ
//
// Usage:
//   ThaanaKeyboard.attach(textareaEl)   — enable
//   ThaanaKeyboard.detach(textareaEl)   — disable
//   ThaanaKeyboard.toggle(textareaEl, bool)

const ThaanaKeyboard = (() => {
    // Source ASCII chars (position = index into PHONETIC string below)
    const FROM = "qwertyuiop[]\\asdfghjkl;'zxcvbnm,./QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?()";

    // Thaana chars at the same index positions (phonetic layout)
    const PHONETIC = "ްއެރތޔުިޮޕ][\\ަސދފގހޖކލ؛'ޒ×ޗވބނމ،./ޤޢޭޜޓޠޫީޯ÷}{|ާށޑﷲޣޙޛޚޅ:\"ޡޘޝޥޞޏޟ><؟)(";

    function _handleKey(e) {
        // Let Ctrl / Meta shortcuts through unchanged
        if (e.ctrlKey || e.metaKey) return;
        const code = e.which || e.keyCode;
        if (!code) return;
        const idx = FROM.indexOf(String.fromCharCode(code));
        if (idx === -1) return;          // not in our map — pass through

        e.preventDefault();
        const ch = PHONETIC[idx];
        const el = e.target;
        const s = el.selectionStart ?? el.value.length;
        const end = el.selectionEnd ?? el.value.length;
        el.value = el.value.slice(0, s) + ch + el.value.slice(end);
        el.selectionStart = el.selectionEnd = s + 1;
        // Notify Alpine.js / vanilla listeners that value changed
        el.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function attach(el) {
        el.addEventListener('keypress', _handleKey);
        el.setAttribute('dir', 'rtl');
        el.setAttribute('lang', 'dv');
    }

    function detach(el) {
        el.removeEventListener('keypress', _handleKey);
        el.removeAttribute('dir');
        el.removeAttribute('lang');
    }

    function toggle(el, enabled) {
        detach(el);
        if (enabled) attach(el);
    }

    return { attach, detach, toggle };
})();
