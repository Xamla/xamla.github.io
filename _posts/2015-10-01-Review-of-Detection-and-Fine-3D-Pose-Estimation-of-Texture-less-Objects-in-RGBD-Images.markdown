---
layout: post
group: review
title:  "Review of Detection and Fine 3D Pose Estimation of Texture-less Objects in RGB-D Images"
date:   2015-10-01 13:51:00
categories: update
mathjax: true
---

**Paper:**
[Detection and Fine 3D Pose Estimation of Texture-less Objects in RGB-D Images](http://cmp.felk.cvut.cz/~hodanto2/darwin/hodan2015detection.pdf)
(T. Hodan, X. Zabulis, M. Lourakis, S. Obdrzalek and J. Matas; Sep 2015)

**Description:** <br />
Pipeline der Objekterkennung und 3D-Posen-Bestimmung:

 \\(\overset{\text{Scanning}}{\overset{\text{window}}{\overset{\text{locations}}{\longrightarrow}}} \ {\small \textbf{Pre-filtering}} \ \longrightarrow \ {\textbf{Hypothesis} \atop \textbf{Generation}} \ \longrightarrow \ {\small \textbf{Verification} \text{ + NMS}} \ \overset{\text{Detections}}{\overset{\text{+ rough 3D}}{\overset{\text{poses}}{\longrightarrow}}} \ {\textbf{Fine 3D Pose} \atop \textbf{Estimation}} \ \overset{\text{Fine 3D}}{\overset{\text{poses}}{\longrightarrow}}\\) 
<br />

* **Pre-filtering:**
Das in diesem Paper verwendete Detektions-Verfahren besteht aus einem "Sliding Window-Ansatz" kombiniert mit einer effizienten kaskardenartigen Auswertung jeder Fensterposition. D.h. es wird eine einfache Vorfilterung durchgeführt, die die meisten Fensterpositionen recht schnell verwirft. 

* **Hypothesis Generation:**
Für jede verbleibende Fensterposition wird mit einem auf Hashing basierten Bewertungs-Verfahren eine Menge von Kandidaten-Templates (Bilder, die das dort abgebildete Objekt aus vielen verschiedenen Blickwinkeln zeigen) identifiziert.

* **Verification:**
Die Kandidaten-Templates werden dann verifiziert, indem verschiedene Merkmale, wie die Objekt-Größe in Relation zur Kameradistanz, u.s.w. zwischen je einem Template und dem betrachteten Bildausschnitt verglichen werden. 

* **Fine 3D Pose Esitmation:**
Schließlich wird die mit jedem ermittelten Template verknüpfte ungefähre Objekt-Pose als Startpunkt für eine stochastische Optimierung (*"Particle Swarm Optimization"*) zur Bestimmung einer exakten 3D-Objekt-Pose verwendet.

**Details zu Punkt 1 (Sliding Window-Ansatz mit Vorfilterung):** <br />
**Sliding Window-Ansatz:** <br />
Die Detektion der Objekte in dem RGB-D Eingangs-Bild basiert auf einem *"Sliding Window-Ansatz"*, der auf verschiedene Bildskalierungen angewandt wird. Die bekannten Objekte werden durch ein Menge von mehreren tausend RGB-D-Template-Bildern fester Größe beschrieben. Diese Template-Bilder erfassen alle möglichen Objekt-Ansichten, wobei die Distanz zur Kamera fest gewählt ist, so dass das Objekt das Template-Bild optimal füllt. <br />
**Vorfilterung:** <br />
Um die Anzahl der zu verarbeitenden Bildpositionen zu verringern, wird jeder Bildausschnitt zunächst anhand seiner Wahrscheinlichkeit, dass er eines der Objekte enthält, bewertet und ggf. verworfen. Dies entspricht einer *"binären Klassifizierung"*, die nur zwischen Objekt und Hintergrund unterscheidet. Die Wahrscheinlichkeitsberechnung basiert dabei auf der Anzahl der "Depth-discontinuity edges" (Tiefenbild-Kanten) innerhalb des Ausschnitts und wird mit Hilfe eines Integral-Bildes berechnet. "Depth-discontinuity edges" treten an Pixeln auf, in denen das Ergebnis des Sobel-Operators (eines einfachen Kantendetektions-Filters) oberhalb eines Schwellwerts liegt. Der Bildausschnitt wird als "Objekt-enthaltend" klassifiziert, wenn die Zahl seiner Tiefenbild-Kanten mindestens 30% der Zahl der Tiefenbild-Kanten des Templates entspricht, welches die wenigsten solcher Kanten enthält. 

**Details zu Punkt 2 (Hypothesen Generierung):** <br />
Für jeden verbleibenden Bildausschnitt, der durch die Vorfilterung gekommen ist, wird nun schnell eine kleine Menge von Kandidaten-Templates (Bilder, die das dort abgebildete Objekt aus vielen verschiedenen Blickwinkeln zeigen) identifiziert. Dieser Schritt kann als *"Multi-Class-Klassifizierung"* angesehen werden, bei der es eine Klasse für jedes Trainings-Template gibt, jedoch keine Hintergrundklasse. Die hier verwendete Methode wählt Kandidaten-Templates aus mehreren trainierten Hashtabellen. Die Hashtabellen sind durch eine Reihe von Messungen indiziert, welche auf ein Template oder Sliding-Window durchgeführt werden. Dazu wird ein reguläres 12x12 Gitter auf dem Trainingstemplate platziert und jedem Gitterpunkt eine Tiefe und eine Oberflächennormale zugeordnet. Von den Gitterpunkten werden nun Triplets abgetastet, die als entsprechende Messung einen Vektor, bestehend aus 2 relativen Tiefenwerten und 3 Normalen liefern.

**Details zu Punkt 3 (Verifikation):** <br />
Da in dem vorherigen Schritt bereits für jeden objekt-enthaltenden Bildausschnitt eine kleine Menge von Kandidaten-Templates identifiziert wurde, kann dieser Schritt nun als eine Menge von *mehreren separaten binären Klassifizierungen* angesehen werden, die jeweils entscheiden, ob der Bildausschnitt das Objekt des entsprechenden Templates enthält, oder nicht. Die Verifikation (auch *"Template-Matching"*) besteht aus einer Reihe von Tests die Folgendes überprüfen: 1.) Objekt-Größe in Relation zur Kameradistanz, 2.) abgetastete Oberflächennormalen, 3.) abgetastete Bildgradienten, 4.) abgetastete Tiefenkarte und 5.) abgetastete Farbe. Ein verifiziertes Template, das also alle Tests bestanden hat, erhält einen abschließenden Punktestand, der beschreibt, wie gut das Template die einzelnen Testkriterien erfüllt hat. Schließlich wird durch **"Non-maxima Suppression"** von allen verfizierten Templates dasjenige ausgewählt, das einen möglichst hohen Punktestand erzielt hat und gleichzeitig eine möglichst große Objektansicht enthält, die möglichst viel zum Szenenverständnis beiträgt.

**Details zu Punkt 4: (Genaue 3D-Posen-Bestimmung)** <br />
Jedes Template ist verknüpft mit einer ungefähren Objekt-Pose aus der Trainingszeit, d.h. einer ungefähren 3D-Rotation und Distanz zum Ursprung des Kamera-Bezugssystems. Diese ungefähre Rotation und Distanz bilden nun die Anfangswerte für eine genaue Berechnung der Translations- und Rotationsparameter, die die Position und Orientierung des Objekts im Raum festlegen. Eingangsparameter für das Verfahren zur genauen 3D-Posen-Bestimmung sind ein Gittermodel des Objekts, die oben genannte initiale Objekt-Pose, ein Tiefenbild und Sensor-spezifische Werte. Ausgabe ist die exakte Objekt-Pose, also die exakte Rotation und Translation des Objekts. 
Zu jedem Gittermodell eines Objekts wird mit Hilfe der Eigenvektoren der Kovarianzmatrix der Knotenpunktmenge eine 3-dimensionale, orientierte Boundingbox vorberechnet.
Kandidaten-Posen werden generiert und dann evaluiert, indem sie verwendet werden um Tiefenbilder aus dem Gittermodell des Objekts zu rendern. Eine Bewertungsfunktion bestimmt die Ähnlichkeit jedes Tiefenbilds mit dem Eingangsbild, indem sie Tiefen-, Kanten- und Orientierungs-Werte vergleicht.
[Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) (PSO) wird verwendet um diese Bewertungsfunktion zu optimieren und diejenige Objekt-Pose zu finden, für die das gerenderte Tiefenbild dem Eingangsbild am ähnlichsten ist. Der Suchraum für die Posen-Bestimmung ist auf eine 6D-Nachbarschaft der Anfangs-Pose eingegrenzt. Jede der Dimensionen des Posen-Suchraums ist beschränkt, was ein *"Such-Hyperrechteck"* definiert, das sich um die Anfangs-Pose dreht. PSO setzt keine Kenntnis über die Ableitung der Zieflunktion vorraus, hängt von nur wenigen Parametern ab und kommt mit nur relativ wenigen Auswertungen der Zielfunktion aus.

Siehe auch die entsprechende [Internetseite des Erstautors](http://cmp.felk.cvut.cz/~hodanto2/) und dessen dort verlinkte weitere Publikationen.