---
layout: post
group: review
title: "Region-based Fully Convolutional Networks"
date:   2016-07-05 13:51:00
categories: update
mathjax: true
---

### Klassifikation:

Der letzte ConvLayer produziert \\(k**2\\) Score Maps für jede Objekt-Kategorie und hat somit eine \\(k**2\dot(C+1)\\)-Ausgabeschicht. Die \\(k**2\\) Score Maps entsprechen einem \\(k\times k\\) Gitter, das relative Positionen beschreibt, wie z.B. "oben linke, oben mittig, ... unten rechts".

Eine positions-sensitive ROI-Pooling-Schicht führt die Informationen aus den Score Maps zusammen und generiert Scores für jede ROI.

#### ROI-Pooling über Gitterzelle (Score Map) \\((i, j)\\)des \\(k\times k\\) Gitters:

\\( r_c(i,j|\Theta) = \sum_{(x,y) \in (i,j)} z_{i,j,c}(x+x_0, y+y_0 | \Theta) / n \\)

\\( r_c(i,j|\Theta) \\): Pooled Response (Voting) in der (i,j)-ten Gitterzelle für die c-te Kategorie <br />
\\( (x,y) \in (i,j) \\): Pixel in der Gitterzelle (i,j) <br />
\\( z_{i,j,c} \\): (i,j)-te Score Map für Kategorie c <br />
