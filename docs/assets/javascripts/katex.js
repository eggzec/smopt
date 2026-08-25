/* Render the maths that pymdownx.arithmatex leaves in the page.
 *
 * mkdocs.yml sets arithmatex to generic mode, so formulas arrive as plain
 * delimited text and KaTeX has to be run over the body. Material loads
 * pages without a full reload, so the work is redone on every navigation
 * via the document$ observable; the plain DOMContentLoaded path is the
 * fallback for when that observable is not present. */

function renderMath(root) {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    // A failed formula should show as source, not blow up the page.
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    renderMath(document.body);
  });
} else {
  document.addEventListener("DOMContentLoaded", function () {
    renderMath(document.body);
  });
}
