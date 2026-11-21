// Общий стиль промежуточных отчётов.
// Сборка из корня проекта: uv run --with typst python reports/build.py

#let accent = rgb("#2a78d6")
#let muted = rgb("#52514e")

#let report(title: "", subtitle: "", date: "", body) = {
  set document(title: title + ". " + subtitle)
  set page(
    paper: "a4",
    margin: (x: 2.2cm, top: 2cm, bottom: 2cm),
    numbering: "1",
    number-align: center,
  )
  set text(font: "PT Sans", size: 11pt, lang: "ru")
  set par(justify: true, leading: 0.62em, spacing: 0.9em)
  set list(indent: 0.4em, spacing: 0.55em)
  set enum(indent: 0.4em, spacing: 0.55em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    v(0.9em)
    text(size: 14.5pt, weight: "bold", it)
    v(0.3em)
  }
  show heading.where(level: 2): it => {
    v(0.6em)
    text(size: 12pt, weight: "bold", it)
    v(0.15em)
  }
  show raw: set text(font: "DejaVu Sans Mono", size: 9.5pt)
  set table(stroke: (x, y) => (
    top: if y == 0 { 0.8pt } else if y == 1 { 0.5pt } else { 0pt },
    bottom: 0.8pt,
  ), inset: (x: 5pt, y: 4pt), align: left)
  show table.cell.where(y: 0): set text(weight: "bold")
  show table: set text(size: 9.5pt)
  show table: set par(justify: false)
  show figure.caption: set text(size: 9.5pt, fill: muted)
  show figure: set block(spacing: 1.2em)
  set figure(supplement: "Рисунок")
  show figure.where(kind: table): set figure(supplement: "Таблица")
  show figure.where(kind: table): set figure.caption(position: top)

  // Шапка
  block(text(size: 18pt, weight: "bold", hyphenate: false, title))
  v(0.2em)
  text(size: 12.5pt, fill: accent, subtitle)
  v(0.1em)
  text(size: 10.5pt, fill: muted, date)
  v(0.4em)
  line(length: 100%, stroke: 0.5pt + muted)
  v(0.3em)

  body
}

// Короткий вывод в рамке
#let takeaway(body) = block(
  width: 100%,
  inset: (x: 10pt, y: 8pt),
  fill: rgb("#eef4fc"),
  stroke: (left: 2.5pt + accent),
  body,
)
