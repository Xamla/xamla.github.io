---
layout: post
group: review
title:  "Review of Freeze-Thaw Bayesian Optimization"
date:   2015-09-23 13:51:00
categories: jekyll update
---

**Paper:**
[Freeze-Thaw Bayesian Optimization](http://arxiv.org/abs/1406.3896)
(K. Swersky, J. Snoek and R. P. Adams; Harvard University and University of Toronto; 16 Jun 2014)

**Code:**
[https://github.com/HIPS/Spearmint](https://github.com/HIPS/Spearmint) ?


**Description:** <br />
Freeze-Thaw = Gefrieren-Tauen

Ziel: Beschleunigung der "Bayesanischen Optimierung" zum Auffinden der optimalen Hyperparameter eines Maschinellen Lernverfahrens.

(Hyperparameter sind z.B. der Regularisierungsparameter für die Bestrafung zu hoher Gewichte, oder die Anzahl der Neuronen eines neuronalen Netzwerks, ...) 

Für die Bayesanische Hyperparameter-Optimierung musste bisher jedes Modell mit festem Hyperparametersetting, komplett austrainiert werden, um die Qualität der entsprechenden Hyperparameter bewerten zu können. Menschliche Experten können meist jedoch recht schnell vorhersagen, ob ein Modell mit den jeweiligen Parametereinstellungen sinnvoll ist, oder nicht. D.h. eine Vorhersage kann i.d.R. bereits erfolgen, wenn das Modell noch nicht komplett, sondern erst ansatzweise trainiert wurde.

Diese Teilinformationen, die schon ansatzweise trainierte Modelle liefern, sollen nun auch innerhalb der Bayesanischen Hyperparameter-Optimierung genutzt werden. D.h. teilweise trainierte Modelle können pausiert (*eingefroren*) werden und später ggf. wieder fortgesetzt (*aufgetaut*) werden. Welche Modelle letztendlich weiterverfolgt werden, wird durch ein Kriterium aus der Informationstheorie entschieden.

Eine wesentliche Annahme für den hier verfolgten Ansatz ist, dass die Zielfunktionswerte während des Trainings in etwa *exponentiell* in Richtung eines unbekannten Endwertes hin abnehmen.

Um solche exponentiell abfallenden Trainingskurven zu charakterisieren, wird ein neuer Kovarianz-Kernel eingeführt, der aus einer unendlich großen Zusammensetzung von exponentiell fallenden Basisfunktionen besteht. Es wird dadurch ein Prior entwickelt, der exponentiell abfallende Funktionen sehr stark begünstigt.

