---
layout: page
title: Paper Library
permalink: /paper-library/
mathjax: true
---

#### [\\(\rightarrow\\) Index of titles](#Title index) <br />


## Papers with description:


#### <a name="Asynchronous Methods for Deep RL"></a>[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

(V. Mnih, A. Puigdomènech B., M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver and K. Kavukcuoglu; 4 Feb 2016)


#### <a name="Generalized Advantage Estimation"></a>[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

(J. Schulman, P. Moritz, S. Levine, M. Jordan and P. Abbeel; 8 Jun 2015)


#### <a name="The Predictron"></a>[The Predictron: End-To-End Learning and Planning](https://arxiv.org/abs/1612.08810)

(D. Silver, H. van Hasselt, M. Hessel, T. Schaul, A. Guez, T. Harley, G. Dulac-Arnold, D. Reichert, N. Rabinowitz, A. Barreto and T. Degris; 20 Jan 2017)


#### <a name="DDPG"></a>[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

(T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver and D. Wierstra; 9 Sep 2015)

*Basierend auf einer früheren Arbeit zu [Deterministic Policy Gradients (DPG)](http://jmlr.org/proceedings/papers/v32/silver14.pdf) von D. Silver et al. und mit Hilfe der Deep Learning Tricks aus dem populären [Deep Q-Networks (DQN)](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)-Paper (bzw. dessen [Vorgänger-Paper](https://arxiv.org/abs/1312.5602)) ist der sogenannte "Deep Deterministic Policy Gradients (DDPG) Algorithmus entstanden, der Probleme mit kontinuierlichen Zustandsräumen und vor allem auch kontinuierlichen Aktionsräumen lösen kann. Der zunächst vielleicht naheliegende Ansatz, den Aktionsraum zu diskretisieren führt leider i.d.R. nicht zum gewünschten Erfolg, da die Aktionen exponentiell mit der Anzahl der Freiheitsgrade (z.B. Gelenke eines Roboterarms) zunehmen ("Curse of Dimonsionality") und eine hinreichend feine Diskretisierung deshalb oftmals nicht praktikabel ist. Der hier vorgestellte DDPG-Algorithmus ist ein "model-free", "off-policy", "Actor-Critic" Algorithmus, der neuronale Netze als Funktionsapproximatoren verwendet und "Reinforcement Learning Policies" in kontinuierlichen Aktionsräumen erlernen kann. In Actor-Critic Algorithmen generiert der "Actor" (der die Struktur der Policy-Funktion  beschreibt) bei gegebenem Zustand eine Aktion und die Critic (die die Struktur der Value-Funktion beschreibt) ermittelt daraufhin zu gegebenem Zustand, sowie der generierten Aktion und dem entsprechenden "Reward" ein Temporal-Difference (TD) Fehlersignal, welches das Lernen des Actors und der Critic in eine entsprechende Richtung steuert. Hier bestehen Actor und Critic aus je einem neuronalen Netz. Das Actor-Netz bekommt den aktuellen Zustand als Input und liefert als Output eine Zahl, die eine Aktion (aus dem kontinuierlichen! Aktionsraum) repräsentiert. Die Ausgabe des Critic-Netzes ist der geschätzte Q-Wert des aktuellen Zustands und der vom Actor-Netz gewählten Aktion. Das [DPG Theorem](http://jmlr.org/proceedings/papers/v32/silver14.pdf) liefert die Update-Regel für die Gewichte des Actor-Netzes. (Silver et al. haben bewiesen, dass der stochastische Policy-Gradient äquivalent zum deterministischen Policy-Gradienten ist. Der Policy-Term in der entsprechenden deterministischen Policy-Gradientenberechnung ist wiederum unabhängig von den Aktionen und es wird nur noch der Gradient der Ausgabe des Critic-Netzes bezüglich seiner Parameter und der Gradient der Ausgabe des Actor-Netzes bezüglich seiner Parameter benötigt.)
Die Gewichte des Critic-Netzes werden mit den TD-Fehlersignal Gradienten aktualisiert. Hierbei werden zwei wesentliche Tricks aus dem DQN-Paper übernommen: 1.) Anwendung der "Experience Replay" Technik: Verwendung eines "Replay-Buffers" um Experiences (Erfahrungen des Agenten bzw. Transitions (s_t,a_t,r_t,s_t+1)) während des Trainings abzuspeichern und später für das Lernen dann zufällig wieder auszuwählen. (Dadurch werden Korrelationen zwischen verschiedenen Trainingsepisoden minimiert.) 2.) Die Netze werden mit Target-Netzen (Ziel-Actor-Netz und Ziel-Q-Netz) trainiert, die konsistente Ziele für die TD-Fehlersignalberechnung liefern und den Lernalgorithmus regularisieren. Zudem wird "Batch-Normalization" angewandt, um die unterschiedlichen Komponenten einer Beobachtung (z.B. Position, Geschwindigkeit, ...) mit unterschiedlichen physikalischen Einheiten, auf einen ähnlichen Wertebereich zu skalieren. Die Policy-Exploration kann bei dem off-policy Algorithmus DDPG unabhängig vom Lernalgorithmus durchgeführt werden. Hier wird ein Rauschen zur Actor-Policy addiert, das mit dem sogenannten "Ornstein-Uhlenbeck Prozess" modelliert wurde. (Der Ornstein-Uhlenbeck prozess modelliert die Geschwindigkeit eines Brown'schen Partikels unter dem Einfluss von Reibung, was zeitlich korrelierte, um 0 zentrierte Werte für das Rauschen liefert.)*

Sehr hilfreicher Blog Post von P. Emami: [Deep Deterministic Policy Gradients (DDPG) in Tensorflow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) <br />


#### <a name="Early Visual Concept Learning"></a>[Early Visual Concept Learning with Unsupervised Deep Learning](https://arxiv.org/abs/1606.05579)

(I. Higgins, L. Matthey, X. Glorot, A. Pal, B. Uria, C. Blundell, S. Mohamed and A. Lerchner; 17 Jun 2016)

siehe auch:  <br />


#### <a name="Double Backprop"></a>[Improving generalization performance using double backpropagation](http://yann.lecun.com/exdb/publis/pdf/drucker-lecun-92.pdf)

(H. Drucker and Y. L. Cun; 6 Nov 1992)

siehe auch:  <br />


#### <a name="Unrolled GANs"></a>[Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163)

(Luke Metz, Ben Poole, David Pfau, Jascha Sohl-Dickstein; 7 Nov 2016)

siehe auch:  <br />


#### <a name="Fast Predictive Image Registration"></a>[Fast Predictive Image Registration](https://arxiv.org/abs/1607.02504)

(X. Yang, R. Kwitt and M. Niethammer; 8 Jul 2016)

siehe auch:  <br />


#### <a name="IcGAN"></a>[Invertible Conditional GANs for image editing](https://arxiv.org/abs/1611.06355)

(G. Perarnau, J. van de Weijer, B. Raducanu and J. M. Álvarez; 19 Nov 2016)

Torch-Implementierung auf GitHub: [https://github.com/Guim3/IcGAN](https://github.com/Guim3/IcGAN) <br />


#### <a name="StackGAN"></a>[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242)

(H. Zhang, T. Xu, H. Li, S. Zhang, X. Huang, X. Wang and D. Metaxas; 10 Dec 2016)

*Das in diesem Paper beschriebene "stacked Generative Adversarial Network" (StackGAN) besteht aus zwei hintereinander geschalteten ("stacked") Generator-Diskriminator-Netzen (GANs) und dient der Generierung von fotorealistischen Bildern aus Testbeschreibungen. Der Prozess der Bildgenerierung wird dabei in zwei einfachere Unterprozesse zerlegt: Das sogenannte "Stage-I GAN" erzeugt in Abhängigkeit von der Textbeschreibung zunächst ein niedrig aufgelöstes Bild mit der groben Objektform und den grundlegenden Objektfarben. Hintergrundbereiche werden aus einem zufälligen "Noise-Vektor", der von einer A-Priori-Verteilung gezogen wurde, erzeugt. Das darauffolgende "Stage-II GAN" generiert nun basierend auf dem zuvor erzeugten niedrig aufgelösten Bild und der entsprechenden Textbeschreibung ein hoch aufgelöstes Bild. Da die groben Farben und Formen des Bildes in der zweiten Phase bereits generiert sind, kann sich das Stage-II GAN nun auf die Ojektdetails und das Erkennen und Ausbessern von Fehlern im Bild konzentrieren. 
Es wird eine sogenannte "Bedingungs-Vermehrungs-Technik" ("conditioning augmentation technique") eingesetzt, um die konditionierenden Variablen (also hier die Textbeschreigungen bzw. deren Einbettungen) für den Generator zu vermehren: Die latenten Variablen werden zufällig von einer Gaußverteilung mit von der Texteinbettung abhängigem Mittelwert und entsprechender Kovarianz gezogen. Dadurch wird die Anzahl der (Text,Bild)-Trainingspaare erhöht und Robustheit gegenüber kleinen Störungen in der eingebetteten Textbeschreibung bewirkt. Um diese Robustheit noch weiter zu stärken und Overfitting zu vermeiden, wird zudem ein Regularisierungsterm in die zu minimierende Generator-Zielfunktion eingefügt, und zwar die Kullback-Leibler Divergenz zwischen zuvor genannter, von der Texteinbettung abhängiger Gaußverteilung und der Standard-Gaußverteilung. 
Das "Stage-II GAN" wird in Abhängigkeit von den zuvor erzeugten, niedrig aufgelösten Bildern und den Gauß'schen latenten Variablen (Texteinbettungen) trainiert. Hierbei wird, anders als bei herkömmlichen GANs, der zufällige "Noise-Vektor" nicht mehr benötigt, weil das zufällige Rauschen schon in den zuvor erzeugten niedrig aufgelösten Bildern steckt. 
Für das Diskriminator-Training zählen echte Bilder und deren zugehörige Textbeschreibung als positive Beispiele, während sowohl echte Bilder mit falscher Textbeschreibungen, als auch synthetische Bilder mit passender Textbeschreibung als Negativ-Beispiele zählen.
Das Training läuft so ab, dass zunächst nur das "Stage-I GAN" für 600 Epochen trainiert wird und danach dann (bei festem Stage-I GAN) das "Stage-II GAN" für 600 Epochen trainiert wird. Für die Optimierung wird "ADAM" verwendet mit einer initialen Lernrate von 2e-4, welche alle 100 Epochen halbiert wird.
Experimentiert wurde auf dem "Caltech-UCSD Bird (CUB)"-Datensatz und dem "Oxford-102 Flower"-Datensatz. Evaluiert wurden die aus Text erzeugten Vogel- und Blumen-Bilder zum Einen durch den sogenannten "Inception Score", der realistisches Aussehen und Vielfalt der generierten Bilder bewertet, und zum Anderen durch Menschen, die zusätzlich den Zusammenhang der generierten Bilder mit ihrer Textbeschreibung beurteilt haben. Auf beiden Datensätzen hat das "StackGAN" verglichen mit anderen Verfahren die besten Scores erzielt. Zudem ist es laut den Autoren das bislang einzige Verfahren, dass realistisch aussehende, hoch aufgelöste Bilder der Größe 256x256 erzeugen kann, die allein von einer Textbeschreibung abhängen.*

Python/Tensorflow-Implementierung auf GitHub: [https://github.com/hanzhanggit/StackGAN](https://github.com/hanzhanggit/StackGAN) <br />
Kommentare auf reddit: [https://www.reddit.com/r/MachineLearning/comments/5i23wt/r_stackgan_text_to_photorealistic_image_synthesis/](https://www.reddit.com/r/MachineLearning/comments/5i23wt/r_stackgan_text_to_photorealistic_image_synthesis/)


#### <a name="GAN Tutorial"></a>[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

(I. Goodfellow; 31 Dec 2016)

Kommentare auf reddit: [https://www.reddit.com/r/MachineLearning/comments/5lpgn8/r_nips_2016_tutorial_generative_adversarial/](https://www.reddit.com/r/MachineLearning/comments/5lpgn8/r_nips_2016_tutorial_generative_adversarial/)


#### <a name="Hyperband"></a>[Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560)

(L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh and A. Talwalkar; 21 Mar 2016)

See also: [https://people.eecs.berkeley.edu/~kjamieson/hyperband.html](https://people.eecs.berkeley.edu/~kjamieson/hyperband.html) <br />
Python-Code auf GitHub: [https://github.com/lishal/hyperband_benchmarks](https://github.com/lishal/hyperband_benchmarks)


#### <a name="DelugeNets"></a>[DelugeNets: Deep Networks with Massive and Flexible Cross-layer Information Inflows](https://arxiv.org/abs/1611.05552)

(J. Kuen, X. Kong and G. Wang; 17 Nov 2016)

*In [ResNets](https://arxiv.org/abs/1512.03385) gibt es die sogenannten "Skip Connections", d.h. direkte Verbindungen zwischen früheren und späteren Schichten, die sehr weit auseinander liegen können. Allerdings sind die "Cross-Layer-Verbindungen" zwischen früheren und späteren Schichten fest und nicht "selektiv", so dass  spätere Schichten keine Möglichkeit haben, die Ausgabekanäle bestimmter früherer Schichten zu priorisieren oder in ihrer Priorität herabzusetzen.
Dicht-verbundene Netze ([DenseNets](https://arxiv.org/abs/1608.06993)) zielen darauf ab, diesen Nachteil von ResNets zu überwinden, indem in den convolutional Layern zusätzlich zu den räumlichen und Feature-Kanal Dimensionen eine extra Dimension berücksichtigt wird: die Schichttiefen-Dimension. Um die damit verbundene große Menge an Parametern zu handhaben, sind DenseNets so konfiguriert, dass sie verglichen mit üblichen CNNs eine wesentlich niedrigere Ausgabe-Breite (12-24 Ausgabe-Kanäle) in jedem Layer haben.
Weil eine zu starke Verringerung der Ausgabe-Breite die Repräsentations-Leistung eines Netzes herabsetzen kann, wird hier eine neue Klasse von CNNs vorgestellt: "DelugeNets". Diese ermöglichen selektive "Cross-Layer-Verbindungen" und haben gleichzeitig eine reguläre Ausgabe-Breite. Inspiriert sind DelugeNets durch "separierbare Convolutions": Die Effizienz von Convolutions kann durch Trennung der beteiligten Dimensionen (Raumdimensionen, Featurekanaldimensionen, Schichttiefendimension) verbessert werden. DelugeNets sind so aufgebaut, dass die Schichttiefen-Dimension unabhängig von den anderen (räumlichen und Featurekanal-) Dimensionen behandelt wird. Die dabei eingesetzte neue Variante von convolutional Layer wird hier "cross-layer depthwise convolutional Layer" genannt und arbeitet ähnlich wie der "depthwise convolutional Layer" (siehe [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)). "Depthwise convolutional Layers" behandeln verschiedene Input-Kanäle als getrennte Gruppen mit unterschiedliche Sets von convolutional Filtern. DelugeNets  erweitern "depthwise convolutional Layer", indem der gleiche Input-Kanal schichtübergreifend (also über viele vorhergehende Schichten hinweg) immer in die gleiche Gruppe einsortiert wird.
Insgesamt werden in DelugeNets Schichten, die auf den gleichen FeatureMap-Dimensionen arbeiten zu einem Block zusammengefasst.
Der Input einer bestimmten Schicht kommt dann von allen vorhergehenden Schichten des gleichen Blocks. Aus den anderen Blöcken kommen keine direkten Informationen in diese Schicht. Für den Übergang zum nächsten Block wird eine "strided spatial Convolution" verwendet um eine FeatureMap passender Dimension zu erzeugen...*

Torch-Implementierung auf GitHub: [https://github.com/xternalz/DelugeNets](https://github.com/xternalz/DelugeNets) <br />
GitXiv-Link: [http://www.gitxiv.com/posts/2xngbbYekco87DySH/delugenets-deep-networks-with-massive-and-flexible-cross](http://www.gitxiv.com/posts/2xngbbYekco87DySH/delugenets-deep-networks-with-massive-and-flexible-cross) <br />
Kommentare auf reddit: [https://www.reddit.com/r/MachineLearning/comments/5l0k6w/r_delugenets_deep_networks_with_massive_and/](https://www.reddit.com/r/MachineLearning/comments/5l0k6w/r_delugenets_deep_networks_with_massive_and/)


#### <a name="pix2pix"></a>[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004v1)

(P. Isola, J.-Y. Zhu, T. Zhou and A. A. Efros; 21 Nov 2016)

*Idee: Entwicklung einer Allgemein-Lösung für die Transformation einer beliebigen Bildrepräsentation in eine beliebige andere Repräsentation (wie z.B. Darstellung als RGB-Bild, als Gradientenfeld, als Karte aller Kanten, etc.). Verwendet werden hierzu "Conditional Generative Adversarial Networks" (cGANs). Diese lernen nicht nur die Transformation des Eingangsbildes in das Ausgabebild, sondern erlernen gleichzeitig eine geeignete Zielfunktion zum Trainieren dieser Transformation. Die zu lernende Zielfunktion versucht, dass Ausgabebild als "real" oder "künstlich erzeugt" zu klassifizieren (Diskriminator), während gleichzeitig ein vom Eingangsbild abhängiges (also bedingtes), generatives Modell trainiert wird (Generator), das diese Zielfunktion minimiert, indem es Ausgaben generiert, die der Diskriminator nur schwer von realen Bildern unterscheiden kann. Durch das Erlernen einer auf die Daten angepassten Zielfunktion können cGANs auf eine Vielzahl verschiedener Tasks angewandt werden, die sehr unterschiedliche Arten von Zielfunktionen voraussetzen. "Conditional GANs" erlernen eine "strukturierte" Zielfunktion, für die die Ausgabepixel bedingt durch das Eingangsbild voneinander abhängen. Hier wird die übliche cGAN-Zielfunktion noch mit einer L1-Distanz-Funktion verknüft, die den Generator nicht nur dazu zwingt, den Diskriminator zu täuschen, sondern eine Ausgabe nahe der "Ground Truth" zu erzeugen. Der Generator besteht hier aus einer "U-Net"-basierten Architektur, d.h. einem Encoder-Decoder-Netz mit "Bottleneck"-Schicht in der Mitte und Skip-Verbindungen zwischen an der Mitte gespiegelten Schichten, was den Austausch von low-level Informationen zwischen Input und Output ermöglicht. Der Diskriminator ist ein convolutional "PatchGAN"-Klassifikator, welcher die Struktur der Ausgabepixel nur innerhalb von Bildausschnitten (Patches) auswertet und ggf. bestraft. Der PatchGAN-Diskriminator klassifiziert also nur kleine Bereiche des Bildes als echt oder künstlich und läuft in einer "convolutional" Art und Weise über das Bild. Für die Erzeugung einer Endauswertung des Diskriminators bezüglich des Gesamtbildes, werden schließlich alle Einzelergebnisse gemittelt. Die Anwendbarkeit von cGANs auf eine Vielzahl verschiedener Aufgabenstellungen und Daten wird hier durch die Bearbeitung 7 verschiedener Transformations-Tasks gezeigt: 1.) Semantische Label <-> Foto (Stadtbilder), 2.) architektonische Label -> Foto (Fassadenbilder), 3.) Karte <-> Luftaufnahme (GoogleMaps-Bilder), 4.) Schwarz-Weiß-Foto -> Farbfoto (ImageNet-Bilder), 5.) Kanten -> Foto (Schuh-Bilder), 6.) Skizze -> Foto (Handtaschen-Bilder), 7.) Tagaufnahme -> Nachaufnahme (Außen-Aufnahmen). Ein Nachteil des vom Eingangsbild abhängigen Generators scheint hier ein eingeschränktes Maß an Generierungsvielfalt zu sein (siehe Zitat am Ende des Abschnitts 2.1, S.3): "Despite the dropout noise, we observe very minor stochasticity in the output of our nets. Designing conditional GANs that produce stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work."*

Torch-Implementierung auf GitHub: [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix) <br />
GitXiv-Link: [http://www.gitxiv.com/posts/jTvyBAZXrz3uEDX4F/image-to-image-translation-with-conditional-adversarial-nets](http://www.gitxiv.com/posts/jTvyBAZXrz3uEDX4F/image-to-image-translation-with-conditional-adversarial-nets)


#### <a name="R-FCN"></a>[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)

(J. Dai, Y. Li, K. He and J. Sun; 20 May 2016)

*Idee: "Fully convolutional" Netz mit positions-sensitiven "Score Maps" und dadurch Beschleunigung des Faster R-CNN-Ansatzes mit seinen deutlich teureren "Pro-Region-Subnetzwerken". Die convolutional Layer führen zu einer Translationsinvarianz, die für die Objekt-Klassifikation zwar hilfreich ist, für die Objekt-Lokalisation jedoch eher hinderlich, da es hierbei auf die genaue Position des Objekts im Bild ankommt. Ohne die nach dem ROI-Pooling-Layer  folgenden mehreren "fully connected" (fc) Layer, bedarf es anderer Mittel wie die hier eingeführten "Score Maps" um genauere Positionsinformationen der Objekte zu gewinnen. Für die Objekt-Klassifikation produziert der letzte convolutional Layer aus z.B. "ResNet-101" für eine ROI (z.B. gewonnen aus dem Region Proposal Net (RPN)) k^2 Score Maps für jede Objekt-Kategorie samt Hintergrund (-> k^2 x (C+1) Score Maps). Die k^2 Score Maps entsprechen einem kxk-Gitter, das relative Positionen in der ROI wie "oben links, oben mittig, ..., unten rechts" beschreibt. Eine positions-sensitive ROI-Pooling-Schicht führt diese Informationen aus den "Score Maps" zusammen und generiert (Objekt-Kategorie-)Scores für jede ROI, die dann in eine Softmax-Funktion eingehen, um damit dann den "Cross-entropy"-Klassifikationsteil der Zielfunktion zu berechnen. Die BoundingBox(BBox)-Regression wird ähnlich wie die Objekt-Klassifikation durchgeführt. Dazu wird neben dem k^2x(C+1)-d Conv-Layer für die Klassifikation ein 4k^2-d Conv-Layer für die Regression verwendet (4k^2C-d Conv-Layer bei Klassen-abhängiger BBox-Regression). Dieser 4k^2-d ConvLayer produziert für eine ROI k^2 Score Maps für jede der 4 BBox-Parameter. Die positions-sensitive ROI-Pooling-Schicht wird auf diese 4k^2 Score Maps angewandt und produziert für jede ROI einen 4k^2-d Vektor, der dann durch "Average Voting" zu einem 4-d Vektor zusammengeführt wird. Dieser 4-d Vektor, der die BBox-Position beschreib, geht dann in den "Smooth-L1"-Regressionsteil der Zielfunktion ein. Insgesamt stellt sich der R-FCN-Ansatz bei gleicher Genauigkeit wie Faster R-CNN als 2.5-20 x schneller heraus (getestet auf den PASCAL VOC- und MS COCO-Datensätzen).*


#### <a name="DCGAN"></a>[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

(A. Radford, L. Metz and S. Chintala; 19 Nov 2015)

*Introduction of Deep Convolutional Generative Adversarial Networks (DCGANs) - a class of CNNs for unsupervised learning. DCGANs are trained on various image datasets and learn a hierarchy of representations from object parts to scenes in both the generator and discriminator. Moreover, the learned features are used for novel tasks as image classification, which demonstrates their applicability as general image representations.*


#### <a name="ALI"></a>[Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)

(V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro and A. Courville; 2 Jun 2016)

(see also: [this blog post](https://ishmaelbelghazi.github.io/ALI/) and [this code](https://github.com/IshmaelBelghazi/ALI) on GitHub, as well as the next paper: "Adversarial Feature Learning")

##### -> see also the following paper:

#### <a name="AFL"></a>[Adversarial Feature Learning](https://arxiv.org/abs/1605.09782)

(J. Donahue, P. Krähenbühl and T. Darrell; 31 May 2016)

*Note: Both papers ("Adversarially Learned Inference" ALI and "Adversarial Feature Learning" AFL) publish the same idea of extending the [generative adversarial network (GAN) approach of I. Goodfellow et al.](http://arxiv.org/abs/1406.2661) by an encoder network that learns rich feature representations. GANs generate realistic-looking synthetic data samples by pitting two neural networks against each other: 1.) A generator network that tries to mimic examples from a training set, that are preferably not distinguishable from the real data and 2.) a discriminator network, that is trained to classify data samples correctly as real data samples and synthetic samples. Thus, on one hand, the discriminator is trained to maximize the probability of correctly classifying real data samples and synthetic samples and on the other hand, the generator is trained to produce samples that fool the discriminator, i.e. that are unlikely to be synthetic according to the discriminator. Now, the idea is to augment this GAN framework by an encoder network which (in contrast to the generator that generates data samples from features) learns rich feature representations from data samples. In the AFL paper this new feature learning framework is called Bidirectional GAN (BiGAN). An alternative approach to BiGAN might be taking the learned intermediate representations from the discriminator as feature representations for related tasks (see also [DCGAN paper](https://arxiv.org/abs/1511.06434)). However, due to the authors of the AFL paper it is not clear that the task of distinguishing between real and generated data entails intermediate representations that are useful as semantic feature representations. In the ALF paper the feature learning capabilities of the BiGAN framework are evaluated by first training it unsupervised and then using the encoder's learned feature representations within supervised learning tasks, such as ImageNet classification and Pascal VOC classification, detection and segmentation. For the ImageNet task each of the three modules (generator, discriminator and encoder) are convnets, where the encoder architecture follows AlexNet. As a result the convolutional filters learned by the encoder have clear Gabor-like structure, similar to those originally reported for the fully supervised AlexNet model. Imagenet classification is evaluated by freezing the first 2-5 layers of the pretrained encoder and randomly reinitializing and training the remainder fully supervised for ImageNet classification. For the Pascal VOC classification, detection and segmentation tasks, the pretrained encoder model is used as the initialization for 1-3 fully connected (FC) layers training, Fast R-CNN (FRCN) training, and Fully Convolutional Network (FCN) training, respectively, replacing the AlexNet model trained fully supervised for ImageNet classification. BiGANs are shown to be competitive with contemporary unsupervised feature learning approaches, like autoencoders. However, their features seem not to be competitive with the features learned from the AlexNet model pretrained fully supervised for ImageNet classification. Finally, the authors emphasize that all presented results constitute only a preliminary exploration of the space of model architectures possible under the BiGAN framework and that they expect results to improve significantly with advancements in generator and discriminator model architectures.*


#### <a name="CNN advances"></a>[Systematic evaluation of CNN advances on the ImageNet](http://arxiv.org/abs/1606.02228)

(D. Mishkin, N. Sergievskiy and J. Matas; 7 Jun 2016)

*Systematic study of recent advances in CNN architectures, image pre-processing and learning rate schedules on the ImageNet classification task. Studies were performed with AlexNet, VGGNet and GoogleNet. In order to accelerate tests, images sizes where smaller (144xN with N ≥ 128) than the commonly used size of 224x224. In all experiments, SGD with momentum 0.9 is used for learning and initial learning rate is set to 0.01. The L2 weight decay for convolutional weights is set to 5e-4 and is not applied to bias. Dropout with probability 0.5 is used before the two last layers. Networks were initialized with [layer-sequential unit-variance](http://arxiv.org/abs/1511.06422) (LSUV), while biases are initialized to zero. Image pixel intensities were scaled by 0.04, after subtracting the mean of BGR pixel values (104 117 124). In the end (conclusions section of the paper) a summary of recommendations is given including non-linearity, colorspace transformation, learning rate decay policy, pooling variant, batch size, design of last network layers, etc. Also a link to their [GitHub code](https://github.com/ducha-aiki/caffenet-benchmark) is give in the paper. In more detail, they suggest to use ELU nonlinearity after convolutional layers without batch normalization (or ReLU with BN). Next, they recommend to use a combination of max and average pooling, where max pooling should be 2x2/2 or overlapping, i.e. 3x3/2 if zero-padding is done. With the choosen settings a linear learning rate decay seems to be best, while it would be of great interest to test other, more promising [optimization methods](http://sebastianruder.com/optimizing-gradient-descent/) than SGD. Moreover, they recommend to learn a colorspace transformation of RGB via a mini-network of 1x1 convolutions placed between the RGB image and the conv1 layer. The best architecture for such a mini-network has been conv1x1x10->conv1x1x3 with VLReLU. Additionally the last fully connected layers should be treated as convolution and predictions should be averaged over all spatial positions via average pooling (...->Pool5->C3->C1->CLF->AvePool->Softmax).*


#### <a name="RCNN-Depth-New"></a>[Cross Modal Distillation for Supervision Transfer](http://arxiv.org/abs/1507.00448)

(S. Gupta, J. Hoffman and J. Malik; 2 Jul 2015)


#### <a name="RCNN-Depth"></a>[Learning Rich Features from RGB-D Images for Object Detection and Segmentation](http://arxiv.org/abs/1407.5736)

(S. Gupta, R. Girshick, P. Arbelaez and J. Malik; 22 Jul 2014)

(see also: [this code](https://github.com/s-gupta/rcnn-depth) on GitHub)


#### <a name="Aligning 3D Models to RGB-D Images of Cluttered Scenes"></a>[Aligning 3D Models to RGB-D Images of Cluttered Scenes](http://people.eecs.berkeley.edu/~sgupta/pdf/rgbd-pose.pdf)
([Inferring 3D Object Pose in RGB-D Images](http://arxiv.org/abs/1502.04652))

(S. Gupta, P. Arbeláez, R. Girshick and J. Malik; 16 Feb 2015)


#### <a name="Context for ObjDet"></a>[Exploring Person Context and Local Scene Context for Object Detection](http://arxiv.org/abs/1511.08177)

(S. Gupta, B. Hariharan and J. Malik; 25 Nov 2015)


#### <a name="Particular object retrieval with integral max-pooling of CNN activations"></a>[Particular object retrieval with integral max-pooling of CNN activations](http://arxiv.org/abs/1511.05879)

(G. Tolias, R. Sicre and H. Jégou; 18 Nov 2015)


#### <a name="Stacked What-Where Auto-encoders"></a>[Stacked What-Where Auto-encoders](https://arxiv.org/abs/1506.02351)

(J. Zhao, M. Mathieu, R. Goroshin and Y. LeCun; 8 Jun 2015)


#### <a name="Stability of SGD"></a>[Train faster, generalize better: Stability of stochastic gradient descent](http://arxiv.org/abs/1509.01240)

(M. Hardt, B. Recht and Y. Singer; 3 Sep 2015)


#### <a name="A MultiPath Network for Object Detection"></a>[A MultiPath Network for Object Detection](http://arxiv.org/abs/1604.02135)

(S. Zagoruyko, A. Lerer, T.-Y. Lin, P. O. Pinheiro, S. Gross, S. Chintala and P. Dollár; 7 Apr 2016)


#### <a name="Semantic Object Parsing"></a>[Semantic Object Parsing with Graph LSTM](http://arxiv.org/abs/1603.07063)

(X. Liang, X. Shen, J. Feng, L. Lin and S. Yan; 23 Mar 2016)


#### <a name="Vector Representation for Objects"></a>[Learning a Predictable and Generative Vector Representation for Objects](http://arxiv.org/abs/1603.08637)

(R. Girdhar, D. F. Fouhey, M. Rodriguez and A. Gupta; 29 Mar 2016)


#### <a name="VQA"></a>[VQA: Visual Question Answering](http://arxiv.org/abs/1505.00468) 

(S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick and D. Parikh; 3 May 2015)

*The task of free-form and open-ended visual question answering (VQA) is introduced. Moreover, a corresponding [challenge](http://www.visualqa.org/challenge.html) is organized, a suitable [dataset](http://www.visualqa.org/download.html) is provided and a [first solution](https://github.com/VT-vision-lab/VQA_LSTM_CNN) to this task is presented. Visual question answering means: Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. Here, both and open-ended answering task and a multiple-choice task are considered. Visual questions selectively target different areas of an image, including background details and underlying context. As a result, a system that succeeds at VQA typically needs a more detailed understanding of the image and complex reasoning than a system producing generic image captions. It requires a potentially vast set of AI capabilities, as e.g. fine-grained recognition, object detection, activity recognition, knowledge base reasoning and commonsense reasoning. VQA is natrually grounded in images requiring the understanding of both text and vision. A rich variety of visual concepts emerge from visual questions and their answers. Several slightly different models for solving the VQA task are presented. The best performing model combines the output of an LSTM with the normalized activations from the last hidden layer of a deep CNN. More precisely, a two hidden layer LSTM is used to encode the text questions and the last hidden layer of a VGGNet (VGG19) is used to encode the images. The image features are moreover l2-normalized. The question and image features are transformed to a common space (each via fully-connected layer + tanh non-linearity) and fused via element-wise multiplication. The result is passed through  a fully connected neural network classifier with 2 hidden layers and 1000 hidden units (dropout 0.5) in each layer with tanh non-linearity, followed by a softmax layer to obtain a distribution over answers. The entire model is learned end-to-end with a cross-entropy loss. VGGNet parameters are frozen to those learned for ImageNet classification and not fine-tuned in the image channel.*


#### <a name="Scene Labeling with 2D LSTM"></a>[Scene Labeling with LSTM Recurrent Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)

(W. Byeon, T. M. Breuel, F. Raue and M. Liwicki; June 2015)


#### <a name="Recurrent CNNs for object recognition"></a>[Recurrent Convolutional Neural Network for Object Recognition](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf)

(M. Liang and X. Hu; 2015)


#### <a name="Recurrent CNNs for scene labeling"></a>[Recurrent Convolutional Neural Networks for Scene Labeling](http://jmlr.org/proceedings/papers/v32/pinheiro14.pdf)

(P. O. Pinheiro and R. Collobert; 2014)


#### <a name="Sparse 3D CNNs"></a>[Sparse 3D convolutional neural networks](http://arxiv.org/abs/1505.02890)

(B. Graham; 12 May 2015)


#### <a name="Spatially-sparse ConvNets"></a>[Spatially-sparse convolutional neural networks](http://arxiv.org/abs/1409.6070)

(B. Graham; 22 Sep 2014)

*Convolutional neural network (CNN) for processing spatially-sparse inputs. Taking advantage of the sparsity allows to train and test deep CNNs more efficiently. Slow max-pooling, i.e. using many layers of 2x2 pooling rather than a smaller number of 3x3 or 4x4 layers, retains more spatial information and is of particular importance for e.g. handwriting recognition. In general, slow pooling is computationally expensive, since the spatial size of the hidden layers reduces more slowly. For sparse input however, this is compensated by the fact that with slow pooling the sparsity is preserved in the early hidden layers. Each hidden variable can be thought of as having a "ground state" corresponding to receiving no meaningful input. In general this ground state is non-zero because of bias terms. When the input array is sparse, only the values of the hidden variables where they differ from their "ground state" have to be calculated. Essentially the convolutional and pooling operations shall be memoized, i.e. they shall be speed up by storing the results of these operations and returning the cached result when the same inputs occur again. Here, for each operation there is only one input (the input corresponding to regions in the ground state) that is expected to be seen repeatedly. Hence, to forward propagate the network, two matrices for each layer of the network are calculated: A "feature matrix" which is a list of row vectors, one for the ground state, and one for each active spatial location in the layer, and a "pointer matrix" which for each spatial location in the convolutional layer stores the number of the corresponding row in the feature matrix. Similar data structures can be used in reverse order for backpropagation. Moreover, a combined representation of two rather different techniques for character recognition is used, here: 1.) Render the pen-strokes at a relatively high resolution (e.g. 40x40) and then use a CNN as classifier, 2.) Draw the character in a low resolution grid (e.g. 8x8) and in each square of the grid calculate an 8-d histogram measuring the amount of movement in each of the 8 compass directions. The combined technique preserves sparsity as the histogram is all zero at sites the pen does not touch. Increasing the number of input features per spatial location only increases the cost of evaluating the first hidden layer, so for sparse input it tends to have a negligible impact on performace. The classification of the CIFAR-10 and CIFAR-100 datasets with a spatially-sparse CNN including network in network (NiN) layers DeepCNiN(5,300) with dropout for the conv3x3-layers and data augmentation via affine transformations produced a test error of 6.28% on CIFAR-10 and 24.30% on CIFAR-100. [DeepCNiN(5,300) means input-conv3x3,300-MP2-conv1x1,300-conv3x3,600-MP2-conv1x1,600-conv3x3,900-MP2-conv1x1,900-conv3x3,1200-MP2-conv1x1,1200-conv3x3,1500-MP2-conv1x1,1500-conv3x3,1800-conv1x1,1800-output with a form of leaky ReLU activation \\(f(x) = x \ \text{for} \ x \geq 0 \ \text{and} \ x/3 \ \text{otherwise}\\) and softmax output.]*


#### <a name="CNNs for object detection and pose estimation"></a>[Deep Exemplar 2D-3D Detection by Adapting from Real to Rendered Views](http://arxiv.org/abs/1512.02497)

(F. Massa, B. Russell and M. Aubry; 8 Dec 2015)

*An end-to-end convolutional neural network (CNN) is applied to 2D-3D [exemplar detection](http://www.cs.cmu.edu/~efros/exemplarsvm-iccv11.pdf) with adaption of natural image features to better align with those of CAD rendered views, which leads to an increase of accuracy and speed. "2D-3D exemplar detection" means that CNN features of a 2D image window containing an object proposal (e.g. obtained via selective search) are compared with the CNN features of rendered views of a library of 3D object CAD models. Then the 3D CAD model view that best matches the style and pose of the input image window is returned. Thus, as result not only the bounding box location of an object in the image is detected, but also its style and 3D pose is estimated. Since there is a domain gap between the appearance of natural images and rendered views of CAD models, an important extension to this method here is the learning of an adaption of natural image features to better align with those of the CAD rendered views before the feature comparison step. To perform this adaption learning, a large training set of aligned natural image and rendered view pairs depicting a similar object ist needed. Since existing datasets are either relatively small or have aligned models that only coarsely approximate object styles, here rendered views of textured object models are composited with natural images, which allows to create a large training set of rendered views of CAD models and composite views with natural background pairs. The adaption learning is incorporated as a module in a CNN-based object detection pipeline: CNN features for a 2D object proposal corresponding to a selective search window are learned, along with CNN features for rendered views of CAD models. Then an adaption of the natural image features to the rendered view features is learned and finally the adpated features are compared with calibrated rendered view features to obtain matching scores for each rendered view. For the adaption learning the following loss is minimized over the transformation function \\(\phi \\), that transforms real image features \\(x_i\\) into CAD rendered view features \\(y_i\\): \\(L(\phi) = - \sum_{i=1,\dots,N} S(\phi(x_i),y_i) + R(\phi)\\), where S is the "squared-cosine similarity" between features, R is a regularization function and the feature transformation \\(\phi\\) is a slight modification of an affine transformation: \\(\phi(x) = ReLU(Ax + b)\\), where "ReLU" is the element-wise maximum over zero. Here the adaptation \\(\phi\\) is implemented in the CNN as a fully-connected layer, followed by a ReLU nonlinearity.*


#### <a name="CNNs for object detection and pose estimation"></a>[Convolutional Neural Networks for joint object detection and pose estimation: A comparative study](http://arxiv.org/abs/1412.7190)

(F. Massa, M. Aubry and R. Marlet; 22 Dec 2014)

*Study of CNNs for jointly detecting objects and estimating their 3D pose. In particular, design and evaluation of different feature representations of oriented objects, and different loss functions that lead a network to learn this representations. One difficulty is that the task requires features with two conflicting properties: The pose of an object has a continuous structure while its category is a discrete variable. Moreover, the amount of training data for the task is very limited. Different approaches on the joint object detection and pose estimation task of the [Pascal3D+ benchmark](http://cvgl.stanford.edu/projects/pascal3d.html) are evaluated using the ["Average Viewpoint Precision" (AVP) metric](http://cvgl.stanford.edu/papers/xiang_wacv14.pdf) for assessing joint detection and viewpoint estimation. A Spatial Pyramid Pooling (SPP) framework (as proposed by He et al.), which applies selective search after the convolutional layers, is used. The CNN is pre-trained on ImageNet (in order to deal with the lack of training data) and only the last layers are adapted and fine-tuned to predict different features identifying the presence of an object and its orientation. Only the azimuth (yaw) angle is considered here, because the pitch and roll angles vary only slightly for the Pascal VOC images. Mainly three different approaches of feature representations with corresponding error functions are analyzed:  1.) A "discrete method", where the subspace of object orientations is a set of discrete points (like the object classes), 2.) A "continuous regression method", where for each class the subspace of object orientations is a circle and background patches are forced to have features far from this circle. Hence, small distances from the features to the circle can be associated to the class (implicit treatment of the classification problem) and the angle predicts the orientation. 3.) An "intermediate method", where in opposition to the other methods, the last network layer is divided in two parts: one part for the classsification and one part for the orientation prediction. The classification layer is followed by a softmax, while the orientation prediction layer is not. The output of the orientation prediction layer can be: a) a point on a circle in 2D (single circle for all classes), or b) a point on a circle in a higher dimensional space (a circle per class), or c) a point on a hyper-cylinder in a higher dimensional space (a hyper-cylinder per class). The "discrete method" provides the best results for orientation prediction and clearly outperforms previous state-of-the-art results, but needs more data to avoid a decrease of detection performance, since it treats each orientation as a separate class. The "intermediate method" (joint classification and continuous orientation estimation) shows that seperating the different orientations improves detection performances (i.e. helps the network to find better representations of the object classes). Unfortunately, with this method the orientation predictions were not as good as those obtained by learning the orientation independently from the classification, which might be caused by a too small weight of the pose loss in the error function, which for the "intermediate method" is composed of a classification term and a pose prediction term.*


#### <a name="Visualize DNNs"></a>[A New Method to Visualize Deep Neural Networks](http://arxiv.org/abs/1603.02518)

(L. M. Zintgraf, T. S. Cohen and M. Welling; 8 Mar 2016)


#### <a name="3D ShapeNets"></a>[3D ShapeNets: A Deep Representation for Volumetric Shape Modeling](http://arxiv.org/abs/1406.5670)

(Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao; 22 Jun 2014)


#### <a name="Hand-Eye Coordination via DL"></a>[Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection](http://arxiv.org/abs/1603.02199)

(S. Levine, P. Pastor, A. Krizhevsky and D. Quillen; 7 Mar 2016)

*New learning-based approach to hand-eye coordination for robotic grasping from monocular images. The method consists of two parts: 1.) A grasp success prediction network \\(g(I_t, v_t)\\), i.e. a deep convolutional neural network (CNN), which as input gets an image \\(I_t\\) and a task-space motion command \\(v_t\\) and as ouput returns the probability of motion command \\(v_t\\) resulting in a successful grasp, and 2.) a servoing function \\(f(I_t)\\), which uses the prediction network to continuously update the robot's motor commands to servo the gripper to a success grasp. By continuously choosing the best predicted path to a successful grasp, the servoing mechanism provides the robot with fast feedback to perturbations and object motion, as well as robustness to inaccurate actuation.
Currently, only vertical pinch grasps are considered , though extensions to other grasp parameterizations would be straightforward.
Importantly, the model does not require the camera to be precisely calibrated with respect to the end-effector, but instead continuously uses visual feedback to determine the spatial relationship between the gripper and graspable objects in the scene.
The grasp prediction CNN was trained with a large dataset of over 800000 grasp attempts collected over the course of two months, using between 6 and 14 robotic manipulators at any given time, with slight differences in camera placement and slight differences in wear and tear on each robot resulting in differences in the shape of the gripper fingers. 
Each grasp \\(i\\) consists of \\(T\\) time steps. At each time step \\(t\\), the robot records the current image \\(I_t^i\\) and the current pose \\(p_t^i\\), and then chooses a direction along which to move the gripper. At the final time step \\(T\\), the robot closes the gripper and evaluates the success of the grasp, producing a label \\(l_i\\). The final dataset contains samples \\((I_t^i, p_T^i − p_t^i, l_i)\\) that consist of the image, a vector from the current pose to the final pose, and the grasp success label. 
The CNN moreover is trained with a cross-entropy loss to match \\(l_i\\), causing the network to output the probability \\(p(l_i = 1)\\).
The servoing mechanism uses the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp. Thereto a "small" optimization on \\(v_t\\) is performed using three iterations of the cross-entropy method (CEM), a simple derivative-free optimization algorithm.
Moreover, the following two heuristics for gripper and robot motion are taken as basis: 1.) The gripper is closed whenever the network predicts that no motion will succeed with a probability that is at least 90% of the best inferred motion. 2.) The gripper is raised off the table whenever the network predicts that no motion has a probability of success that is less than 50% of the best inferred motion.
During data collection, grasp success was evaluated using two methods: 1.) The position reading on the gripper is greater than 1cm, indicating that the fingers have not closed fully (only suitable for thick objects). 2.) The images of the bin containing the objects recorded before and after a drop differ, indicating that there has somenthing been in the gripper ("drop test").
Finally, the presented method has been tested to be more robust to perturbations as movement of objects in the scene and variability in actuation and gripper shape than an "open-loop approach" (without continuous feedback). Moverover, grasps automatically were adapted to the different material properties of the objects and even challenging (e.g. flat) objects could be grasped.*

([*Review*]({{ site.baseurl }}/update/2016/03/15/Review-of-Learning-Hand-Eye-Coordination-via-DL.html))


#### <a name="Deconv"></a>[Learning Deconvolution Network for Semantic Segmentation](http://cvlab.postech.ac.kr/research/deconvnet/)

(H. Noh,	S. Hong	and B. Han; 17 May 2015)


#### <a name="Learning to Segment Object Candidates"></a>[Learning to Segment Object Candidates](http://arxiv.org/abs/1506.06204)

(P. O. Pinheiro, R. Collobert and P. Dollar; 20 Jun 2015)

*ConvNet approach for generating object proposals for the object detection task. Main difference to "Faster R-CNN": The method presented here (called "DeepMask") generates segmentation proposals instead of less informative bounding box proposals. The core of this approach is a ConvNet which jointly predicts a segmentation mask given an input patch and assigns an object score corresponding to how likely the patch is to contain an object. A large part of the network is shared between those two tasks: only the last few network layers are specialized for separately outputting a mask and score prediction. The model is trained by optimizing a cost function that targets both tasks simultaneously. In detail, a VGG-A ConvNet architecture (initialized with pre-trained ImageNet classification parameters and consisting of 3x3 convolutions, ReLUs and 2x2 max pooling layers) is used, where the final fully connected layers and the last max-pooling layer are removed, because the spatial information provided in the convolutional feature maps is needed for inferring segmentation masks. The branch of the network dedicated to segmentation is composed of a single 1x1 convolution with ReLU and a classification layer, consisting of several pixel classifiers. The classification layer moreover is decomposed into two linear layers with no non-linearity in between (a low-rank variant of using fully connected linear classifiers, reducing the number of network parameters while allowing each pixel classifier to use information from the entire feature map). The branch of the network dedicated to scoring is composed of a 2x2 max-pooling and two fully connected layers with ReLUs and dropout. The loss function is a sum of binary logistic regression losses, one for each location of the segmenation network and one for the object score. An alternation between backpropagation through the segmenation and scoring branch is performed. Generalization capabilities of the model are demonstrated by testing it on object categories not seen during training. For this, segmentation training with only positive scored objects is critical, since this way the network attempts to generate a segmentation mask at every patch, even if no known object is present. During full image inference the model is densely applied at multiple locations and scales. Since all computations can be computed convolutionally, the full image inference procedure is still efficient. For the implementation of all experiments Torch7 has been used.*


#### <a name="Deep Generative Image Models"></a>[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](http://arxiv.org/abs/1506.05751)

(E. Denton, S. Chintala, A. Szlam and R. Fergus; 18 Jun 2015)


#### <a name="Inside-Outside Net"></a>[Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks](http://arxiv.org/abs/1512.04143)

(S. Bell, C. L. Zitnick, K. Bala and R. Girshick; 14 Dec 2015)


#### <a name="ResNet with pre-activation"></a>[Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027)

(K. He, X. Zhang, S. Ren and J. Sun; 16 Mar 2016)

*Proposal of a rearranged residual building block with “pre-activation”. More precisely, placing the rectified linear unit (ReLU) and batch normalization (BN) before the weight layers instead of after, is shown to further improve the results of the deep residual network.*


#### <a name="Deep Residual Learning for Image Recognition"></a>[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

(K. He, X. Zhang, S. Ren and J. Sun; 10 Dec 2015) 

*Winner of the [ILSVRC 2015](http://image-net.org/challenges/LSVRC/2015/) object detection and image classification and localization tasks. Neural networks with depth of over 150 layers are used together with a "deep residual learning" framework that eases the optimization and convergence of extremely deep networks. The localization and detection systems are in addition based on the ["Faster R-CNN"](http://arxiv.org/abs/1506.01497) system of S. Ren at al.*

Interesting links on residual nets: 

* [Slides of their talk at the ICCV 2015](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)
* [Original implementation on GitHub](https://github.com/KaimingHe/deep-residual-networks)
* [Torch implementation on GitHub](https://github.com/facebook/fb.resnet.torch)
* [Torch blog post about residual nets](http://torch.ch/blog/2016/02/04/resnets.html)
* [GitXiv entry](http://gitxiv.com/posts/LgPRdTY3cwPBiMKbm/deep-residual-learning-for-image-recognition)
* [Short description on Quora](https://www.quora.com/How-does-deep-residual-learning-work)


#### <a name="Learning to think"></a>[On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models](http://arxiv.org/abs/1511.09249)

(J. Schmidhuber; 30 Nov 2015)


#### <a name="Multi-scale video prediction"></a>[Deep multi-scale video prediction beyond mean square error](http://arxiv.org/abs/1511.05440)

(M. Mathieu, C. Couprie and Y. LeCun; 17 Nov 2015)


#### <a name="Inception2"></a>[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

(C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna; 2 Dec 2015)


#### <a name="ELUs"></a>[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289)

(D.-A. Clevert, T. Unterthiner and S. Hochreiter; 23 Nov 2015)


#### <a name="HyperparamOpt"></a>[Gradient-based Hyperparameter Optimization through Reversible Learning](http://arxiv.org/abs/1502.03492)

(D. Maclaurin, D. Duvenaud and R. P. Adams; 11 Feb 2015)

[Interessante Rezension zu dem Paper](https://www.evernote.com/shard/s189/sh/eb8503b0-f63d-49b7-b51f-747b7e10e69e/4cabf65c28de7f388417c0c4fba71c29)


#### <a name="SpeedLearning"></a>[Speed learning on the fly](http://arxiv.org/abs/1511.02540)

(P.-Y. Massé and Y. Ollivier; 8 Nov 2015)

[Interessante Rezension zu dem Paper](https://www.evernote.com/shard/s189/sh/b962600a-48cd-4b6c-98d5-17874f011d3f/b434df41c0e343cbfcd4f77c8148c500)


#### <a name="ReNet"></a>[ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](http://arxiv.org/abs/1505.00393)

(F. Visin, K. Kastner, K. Cho, M. Matteucci, A. Courville and Y. Bengio; 3 May 2015)


#### <a name="Weights and Connections Learning"></a>[Learning both Weights and Connections for Efficient Neural Networks](http://arxiv.org/abs/1506.02626)

(S. Han, J. Pool, J. Tran and W. J. Dally; 8 Jun 2015)


#### <a name="RGB-D Human Attribute Classification"></a>[Real-Time Full-Body Human Attribute Classification in RGB-D Using a Tessellation Boosting Approach](https://cld.pt/dl/download/5b6bf977-9bf2-403b-b88b-20bbfcc7ba4c/MyPapers08/pyc1940617492.pdf)

(T. Linder and K. O. Arras; Sept 2015)


#### <a name="SimTrack"></a>[SimTrack: A Simulation-based Framework for Scalable Real-time Object Pose Detection and Tracking](http://www.karlpauwels.com/downloads/iros_2015/Pauwels_IROS_2015.pdf)

(K. Pauwels and D. Kragic; Sept 2015) <br />


#### <a name="FaceNet"></a>[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)

(F. Schroff, D. Kalenichenko and J. Philbin; 2015) <br />
(Github-Code: https://github.com/cmusatyalab/openface)


#### <a name="Memory foam"></a>[“Memory foam” approach to unsupervised learning](http://arxiv.org/abs/1107.0674)

(N. B. Janson and C. J. Marsden; 4 Jul 2011)


#### <a name="Grid LSTM"></a>[Grid Long Short-Term Memory](http://arxiv.org/abs/1507.01526)

(N. Kalchbrenner, I. Danihelka and A. Graves; 6 Jul 2015)


#### <a name="DeepCamera"></a>[DeepCamera: A Unified Framework for Recognizing Places-of-Interest based on Deep ConvNets](http://dl.acm.org/citation.cfm?id=2806620)

(P. Peng, H. Chen, L. Shou, K. Chen, G. Chen and C. Xu; 2015)


#### <a name="Look Twice"></a>[Look and Think Twice: Capturing Top-Down Visual Attention with Feedback Convolutional Neural Network](http://web.engr.illinois.edu/~xliu102/assets/papers/iccv2015_fnn.pdf)

(C. Cao, X. Liu, Y. Yang, Y. Yu, J. Wang, Z. Wang, Y. Huang, L. Wang, C. Huang, W. Xu, D. Ramanan and T. S. Huang; 2015)

*Einführung von "Feedback-Loops" in vorwärtsgerichteten ConvNets, die den Aktivierungsstatus der inneren Neuronen gemäß den Ergebnissen der oberen Schicht (z.B. der Ausgabe der Klassen-Neuronen) optimieren. 
D.h. wenn ein semantischer Stimulus der oberen Schicht wie z.B. "Panda" gegeben ist, werden durch eine iterative Optimierung in den Feedback-Loops nur solche Neuronen der inneren Schichten aktiviert, welche mit dem Konzept "Panda" verbunden sind, so dass auch nur solche Regionen in der Visualisierung erfasst werden, die für das Konzept "Panda" typisch ("salient") sind. 
Genauer wird das Verhalten der ReLU- und Max-Pooling-Schichten (beide enthalten die "Max-Funktion") als eine Menge von binären Aktivierungsvariablen \\(z \in\\) {0, 1} sogenannten "Gates", die durch die Eingabe gesteuert werden, neu interpretiert. Dadurch können nach dem Sammeln von Informationen in der vorwärtsgerichteten Phase, Singale mit nur geringem Beitrag zur Entscheidungsfindung bzw. mit irrelevanter Information für ein bestimmtes Label in der rückwärtsgerichteten Phase eliminiert werden. Der Feedback-Mechanismus wird als Optimierungsproblem formuliert, der die Zahl der aktiven Neuronen minimiert und dabei den "Score" der Zielklasse maximiert, indem die binären Neuronen-Aktivierungen \\(z\\) jedes inneren Neurons entsprechend angepasst werden. Um dieses Optimierungsproblem in polynomialer Zeit lösen zu können, werden die ganzzahligen Nebenbedingungen \\(z \in\\) {0, 1} durch die relaxierten Nebenbedingungen \\(0 \leq z \leq 1\\) ersetzt ([LP-Relaxation](https://de.wikipedia.org/wiki/LP-Relaxation)).
Durch die Feedback-Loops wird der menschliche Prozess der visuellen Aufmerksamkeit, nämlich das Fokussieren einzelner Objekte nach einem ersten flüchtigen Blick auf das Gesamtbild, nachgeahmt. Der erste flüchtige Blick wird hier durch eine "weakly-supervised" Objekt-Lokalisierung nachgebildet, um eine erste grobe Schätzung der Objektregionen zu erhalten. Diese "weakly-supervised" Objekt-Lokalisierung wird mittels "Salience Maps" realisiert: Bei gegebenem Eingangsbild und entsprechender "Salience Map" für eine bestimmte Klasse, wird eine Objekt-Segmentierungsmaske durch eine einfache Schwellwertbildung bestimmt. Dann wird die enganliegenste Bounding Box um diese Segmentierungsmaske ermittelt. Das Fokussieren wird schließlich realisiert, indem das Netzwerk auf diese Bounding Box-Regionen fokussiert wird, um die finale Klassifizierung durchzuführen. Dieses Fokussieren verbessert die Klassifizierungsgenauigkeit besonders für sehr kleine Objekte.*


#### <a name="DeepBox"></a>[DeepBox: Learning Objectness with Convolutional Networks](http://arxiv.org/abs/1505.02146)

(W. Kuo, B. Hariharan and J. Malik; 8 May 2015)


#### <a name="Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders"></a>[Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders](http://arxiv.org/abs/1509.06113)

(C. Finn, X. Y. Tan, Y. Duan, T. Darrell, S. Levine and Pieter Abbeel; 21 Sep 2015)


#### <a name="RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features"></a>[RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features](http://www.ais.uni-bonn.de/papers/ICRA_2015_Schwarz_RGB-D-Objects_Transfer-Learning.pdf)

(M. Schwarz, H. Schulz and S. Behnke; May 2015)


#### <a name="Learning Compound Multi-Step Controllers under Unknown Dynamics"></a>[Learning Compound Multi-Step Controllers under Unknown Dynamics](http://rll.berkeley.edu/reset_controller/reset_controller.pdf)

(W. Han, [S. Levine](http://www.eecs.berkeley.edu/~svlevine/) and [P. Abbeel](http://www.cs.berkeley.edu/~pabbeel/); 2015)

*Vorgestellt wird ein Reinforcement Learning-Verfahren zum Trainieren einer Verkettung von mehreren Steuerungseinheiten zur Steuerung eines Roboters. Zwei wichtige Voraussetzungen hierfür sind die Berücksichtigung der Zustandsverteilungen, die durch vorhergehende Steuerungseinheiten in der Kette verursacht wurden, und das automatische Trainieren von Reset-Steuerungseinheiten, die das System für jede Episode in seinen Anfangszustand zurücksetzen. Der Anfangszustand jeder Steuerungseinheit wird durch die ihr vorausgehende Steuerungseinheit bestimmt, wodurch sich ein nicht-stationäres Lernproblem ergibt. Es wird gezeigt, dass ein von den Autoren kürzlich entwickeltes Reinforcement Learning-Verfahren, das linear-Gauß'sche Steuerungen mit gelernten, lokalen, linearen Modellen trainiert, derartige Probleme mit nicht-stationären initialen Zustandsverteilungen lösen kann, und dass das gleichzeitige Trainieren von vorwärts gerichteten Steuerungseinheiten zusammen mit entsprechenden Reset-Steuerungseinheiten die Trainingszeit nur minimal erhöht. Außerdem wird das hier vorgestellte Verfahren anhand einer komplexen "Werkzeug-Verwendungs-Aufgabe" demonstriert. Die Aufgabe besteht aus sieben verschiedenen Stufen ("Episoden") und setzt die Verwendung eines Spielzeug-Schraubenschlüssels voraus, um eine Schraube einzudrehen. Abschließend wird gezeigt, dass das hier vorgestellte Verfahren mit ["guided Policy Search"](http://graphics.stanford.edu/projects/gpspaper/index.htm) kombiniert werden kann, um nichtlineare neuronale Netzwerk-Steuerungseinheiten für eine "Greif-Aufgabe" mit beachtlicher Variation in der Zielposition zu trainieren.*

([*Review*]({{ site.baseurl }}/update/2015/10/05/Review-of-Learning-Compound-Multi-Step-Controllers-under-Unknown-Dynamics.html))


#### <a name="Learning Descriptors for Object Recognition and 3D Pose Estimation"></a>[Learning Descriptors for Object Recognition and 3D Pose Estimation](http://arxiv.org/abs/1502.05908)

(P. Wohlhart and V. Lepetit; 20 Feb 2015)

*Verwendung von ConvNets um Descriptoren von Objektansichten zu erhalten, die sowohl die Identität als auch die 3D-Pose des Objekts effizient erfassen. Anstelle von vollständigen Testbildern (vgl. nachfolgendes Paper von T. Hodan) werden hier nur Regionen, die das zu identifizierende Objekt enthalten, als Input für das ConvNet verwendet. Das ConvNet wird trainiert, indem einfache Ähnlichkeits- und Unähnlichkeits-Bedingungen zwischen den Descriptoren erzwungen werden. Genauer soll die Euklidische Distanz zwischen Descriptoren verschiedener Objekte groß sein, während die Euklidische Distanz zwischen Descriptoren gleicher Objekte die Ähnlichkeit ihrer Posen widerspiegeln soll. Weil hier die Ähnlichkeit zwischen den resultierenden Descriptoren durch die Euklidische Distanz evaluiert wird, können skalierbare Nächste-Nachbarn-Suchverfahren verwendet werden, um eine große Anzahl von Objekten unter einer großen Bandbreite von Posen effizient zu bearbeiten. Der gelernte Descriptor generalisiert gut auf unbekannte Objekte. Das Verfahren kann sowohl mit RGB, also auch RBG-D Bildern arbeiten und schlägt moderne Verfahren auf dem öffentlichen Datensatz von [S. Hinterstoisser et al.](http://campar.in.tum.de/Main/StefanHinterstoisser) (http://campar.in.tum.de/pub/hinterstoisser2011pami/hinterstoisser2011pami.pdf)*


#### <a name="Detection and 3D Pose Estimation"></a>[Detection and Fine 3D Pose Estimation of Texture-less Objects in RGB-D Images](http://cmp.felk.cvut.cz/~hodanto2/darwin/hodan2015detection.pdf)

([T. Hodan](http://cmp.felk.cvut.cz/~hodanto2/), X. Zabulis, M. Lourakis, [S. Obdrzalek](http://cmp.felk.cvut.cz/~xobdrzal/) and [J. Matas](http://cmp.felk.cvut.cz/~matas/); Sep 2015)

*Das in diesem Paper verwendete Detektions-Verfahren besteht aus einem "Sliding Window-Ansatz" kombiniert mit einer effizienten kaskardenartigen Auswertung jeder Fensterposition. D.h. es wird eine einfache Vorfilterung durchgeführt, die die meisten Fensterpositionen mittels eines einfachen Wahrscheinlichkeitstests anhand der Anzahl der Tiefenbild-Kanten recht schnell verwirft. Für jede verbleibende Fensterposition wird mit einem auf Hashing basierten Bewertungs-Verfahren eine Menge von Kandidaten-Templates (Bilder, die das dort abgebildete Objekt aus vielen verschiedenen Blickwinkeln zeigen) identifiziert. Dies macht die Berechnungskomplexität des Verfahrens weitgehend unabhängig von der Anzahl der bekannten Objekte. Die Kandidaten-Templates werden dann verifiziert, indem verschiedene Merkmale, wie die Objekt-Größe in Relation zur Kameradistanz zwischen je einem Template und dem betrachteten Bildausschnitt verglichen werden. Jedes Template ist verknüpft mit einer Pose aus der Trainingszeit, d.h. einer 3D Rotation und einer Distanz zum Ursprung des Kamera-Bezugssystems. Schließlich wird die mit jedem ermittelten Template verknüpfte ungefähre Objekt-Pose als Startpunkt für eine stochastische [Partikelschwarmoptimierung (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization) zur Bestimmung einer exakten 3D-Objekt-Pose verwendet.*

([*Review*]({{ site.baseurl }}/update/2015/10/01/Review-of-Detection-and-Fine-3D-Pose-Estimation-of-Texture-less-Objects-in-RGBD-Images.html))


#### <a name="Atari1"></a>[Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)

(V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra and M. Riedmiller; 19 Dec 2013)


#### <a name="Atari2"></a>[Human-level control through deep reinforcement learning](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D)
(bzw. http://rdcu.be/cdlg)

(V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg and D. Hassabis; 25 Feb 2015)

([Kritik von J. Schmidhuber](https://plus.google.com/100849856540000067209/posts/eLQf4KC97Bs))


#### <a name="Freeze-Thaw Bayesian Optimization"></a>[Freeze-Thaw Bayesian Optimization](http://arxiv.org/abs/1406.3896)

(K. Swersky, J. Snoek and R. P. Adams; 16 Jun 2014)

*Beschleunigung der "Bayesanischen Optimierung" zum Auffinden der optimalen Hyperparameter eines Maschinellen Lernverfahrens. Eine Vorhersage über die Güte der Parametereinstellung kann i.d.R. bereits erfolgen, wenn das Modell erst ansatzweise trainiert wurde. "Freeze-Thaw" (Gefrieren-Tauen): Das Modell wird deshalb nicht mehr für jede Parametereinstellung bis zum Ende austrainiert, sondern nach Bedarf pausiert ("eingefrohren") und später ggf. wieder fortgesetzt ("aufgetaut"). Eine wesentliche Annahme für den hier verfolgten Ansatz ist, dass die Zielfunktionswerte während des Trainings in etwa exponentiell in Richtung eines unbekannten Endwertes hin abnehmen. Deshalb wird ein "Prior" entwickelt, der exponentiell abfallende Funktionen besonders begünstigt.* <br />

([*Review*]({{ site.baseurl }}/update/2015/09/23/Review-of-Freeze-Thaw-Bayesian-Optimization.html))


#### <a name="Recurrent Spatial Transformer Networks"></a>[Recurrent Spatial Transformer Networks](http://arxiv.org/abs/1509.05329)

(S. Kaae Sønderby, C. Kaae Sønderby, L. Maaløe and O. Winther; 17 Sept 2015)

*Einbau des von M. Jaderberg und Co. entwickelten "Spatial Transformer"-Moduls (SPN) (s.u.) in ein rekurrentes neuronales Netzwerk (RNN). Verwendung einer affinen Transformation und einer bilinearen Interpolation innerhalb des Spatial Transformer-Moduls. Das rekurrente neuronale Netz mit Spatial Transformer-Modul (RNN-SPN) ist, im Gegensatz zum Feedforward Netz mit Spatial Transformer-Modul (FNN-SPN), imstande jedes Element einer Sequenz (z.B. jede Ziffer einer Hausnummer) einzeln zu behandeln und deshalb für die Klassifikation von Sequenzen besser geeignet.*


#### <a name="Spatial Transformer Networks"></a>[Spatial Transformer Networks](http://arxiv.org/abs/1506.02025)

(M. Jaderberg, K. Simonyan, A. Zisserman and K. Kavukcuoglu; 5 Jun 2015)

*Einführung eines differenzierbaren "Spatial Transformer"-Moduls, das in bestehende ConvNet-Architekturen eingefügt werden kann und es so ermöglicht, die Featuremaps eines ConvNets während des Trainings räumlich zu transformieren, also eine räumliche Invarianz gegenüber Translation, Rotation, Skalierung, etc. automatisch mitzutrainieren. Die Spatial Transformer-Module können mittels gewöhnlicher Back-Propagation trainiert werden, so dass ein End-to-End-Training des gesamten Netzwerks möglich bleibt. Der Aufbau des Spatial Transformer-Moduls lässt sich in drei Teile unterteilen: 1.) Ein Lokalisierungs-Netzwerk \\(f_\text{loc}\\), welches als Eingabe eine Feature Map bekommt und die Parameter der räumlichen Transformation, die darauf angewandt werden soll, ausgibt. 2.) Ein Gitter-Generator \\(f_\text{gridGen}\\), der die vorhergesagten Transformations-Parameter verwendet um ein Sampling-Gitter zu erzeugen, welches aus einer Menge von Punkten besteht, auf der die Eingangs-Feature Map ausgewertet werden soll. 3.) Ein Sampler \\(f_\text{sampler}\\), der die Eingangs-Feature Map und das Sampling-Gitter verwendet, um aus der Eingangs-Feature Map, ausgewertet an den Gitterpunkten des Sampling-Gitters, die Ausgabe-Feature Map zu erzeugen. 
\\(\small (f_\text{loc}(U) = \theta \ \rightarrow \ f_\text{gridGen}(\theta, G) = \tau_\theta(G) \ \rightarrow \ f_\text{sampler}(U, \tau_\theta(G))=V)\\)*


#### <a name="R-CNN minus R"></a>[R-CNN minus R](http://arxiv.org/abs/1506.06981)

(K. Lenc and A. Vedaldi; 23 Jun 2015)

*Alternative zum "Faster R-CNN"-Verfahren von S. Ren und Co.
<!-- S. Ren und Co verwenden die "convolutional Features" dazu ein neues, effizientes Kandidaten-Regionen-Generierungs-Schema zu entwickeln. -->
Die Bild-abhängige Generierung der Kandidaten-Regionen wird komplett entfernt ("R-CNN minus R"). Stattdessen wird eine Bild-unabhängige Liste von Kandidaten-Regionen verwendet und darauf gesetzt, dass das ConvNet im Nachhinein eine akkurate Lokalisierung durchführt. Um eine Liste von Bild-unabhängigen Kandidaten-Regionen zu konstruieren, wird zunächst die Verteilung der Bounding-Boxen in einem repräsentativen Object Detection-Datensatz, hier dem PASCAL VOC 2007-Datensatz, studiert. Die Boxen tendieren dazu, nahe dem Bildzentrum aufzutreten und das Bild zu füllen (\\(\rightarrow\\) "Ground Truth-Verteilung"). 
Die Abtastung eines "Sliding Window"-Verfahrens wird nun so modifiziert, dass die Bounding Box-Verteilung am Ende der PASCAL VOC 2007-Bounding Box-Verteilung entspricht. Dies wird erreicht, indem n K-means Kluster anhand der Bounding Boxen aus dem PASCAL VOC 2007-Datensatz berechnet werden.
Wie erwartet, entspricht die resultierende Verteilung gut der Ground Truth-Verteilung, sogar für eine kleine Menge von n = 3000 Kluster-Zentren. 
Die so erzeugte Bild-unabhängige Menge von Kandidaten-Regionen kombiniert mit einem ConvNet-basierten Bounding Box-Regressor, resultiert in einem ähnlich guten (nur wenig schlechteren) und sehr schnellen Object-Detector verglichen mit dem Faster R-CNN Object Detector.*


#### <a name="Faster R-CNN"></a>[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)

(S. Ren, K. He, R. Girshick and J. Sun; 4 Jun 2015)


#### <a name="Fast R-CNN"></a>[Fast R-CNN](http://arxiv.org/abs/1504.08083) 

(R. Girshick; 30 Apr 2015)


#### <a name="R-CNN"></a>[Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) 

(R. Girshick, J. Donahue, T. Darrell and J. Maliket; 11 Nov 2013)


#### <a name="SPPNet"></a>[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](http://arxiv.org/abs/1406.4729)

(K. He, X. Zhang, S. Ren and J. Sun; 18 Jun 2014)


#### <a name="ZFNet"></a>[Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)

(M. D. Zeiler and R. Fergus; 12 Nov 2013)


#### <a name="VGG"></a>[Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556)

(K. Simonyan and A. Zisserman; 4 Sep 2014)


#### <a name="GoogLeNet"></a>[Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)

(C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke and A. Rabinovich; 17 Sep 2014)


#### <a name="Fractional Max-Pooling"></a>[Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

(B. Graham; 18 Dec 2014)


#### <a name="NiN"></a>[Network In Network](http://arxiv.org/abs/1312.4400)

(M. Lin, Q. Chen and S. Yan; 16 Dec 2013)


#### <a name="AlexNet"></a>[ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-)

(A. Krizhevsky, I. Sutskever and G. E. Hinton; 2012)


#### <a name="Deep Learning using Linear Support Vector Machines"></a>[Deep Learning using Linear Support Vector Machines](http://arxiv.org/abs/1306.0239)

(Y. Tang; 2 Jun 2013)

Winner of the [ICML 2013 Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) hosted on Kaggle


#### <a name="Deep Learning in Neural Networks: An Overview"></a>[Deep Learning in Neural Networks: An Overview](http://arxiv.org/abs/1404.7828)

(J. Schmidhuber; 30 Apr 2014)


#### <a name="Traffic Sign Detection"></a>[A robust, coarse-to-ﬁne trafﬁc sign detection method](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6706812&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D6706812)

(G. Wang, G. Ren, Z. Wu, Y. Zhao and L. Jiang; 2013)

Winner (1st place) of the German Traffic Sign Detection Benchmark (GTSDB)


#### <a name="Deep Sparse Rectifier Neural Networks"></a>[Deep Sparse Rectifier Neural Networks](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)

(X. Glorot, A. Bordes and Y. Bengio; 2011)


#### <a name="Street View House Numbers"></a>[Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/abs/1312.6082)

(I. J. Goodfellow, Y. Bulatov, J. Ibarz, S. Arnoud and V. Shet; 20 Dec 2013)


#### <a name="Large Scale Distributed Deep Networks"></a>[Large Scale Distributed Deep Networks](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks)

(J. Dean, G. S. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. A. Ranzato, A. Senior, P. Tucker, K. Yang and A. Y. Ng; 2012)


#### <a name="Traffic Sign Classification"></a>[Multi-column deep neural network for traffic sign classification](http://xa.yimg.com/kq/groups/14962965/499730234/name/NN_PAPER.pdf)

(D. Ciresan, U. Meier, J. Masci and J. Schmidhuber; 2012)


#### <a name="Bias-Variance Tradeoff"></a>[Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)

(S. Fortmann-Roe (developer of [Insight Maker](http://insightmaker.com)); June 2012)


#### <a name="Difficulty of Training DNN"></a>[Understanding the difﬁculty of training deep feedforward neural networks](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)

(X. Glorot and Y. Bengio; 2010)


#### <a name="Initialization and Momentum"></a>[On the importance of initialization and momentum in deep learning](http://jmlr.org/proceedings/papers/v28/sutskever13.pdf)

(I. Sutskever, J. Martens, G. Dahl and G. Hinton; 2013)


#### <a name="MNIST"></a>[Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition](http://arxiv.org/abs/1003.0358)

D. C. Ciresan, U. Meier, L. M. Gambardella and J. Schmidhuber; 1 Mar 2010)


#### <a name="High-level Features via Unsupervised Learning"></a>[Building High-level Features Using Large Scale Unsupervised Learning](http://arxiv.org/pdf/1112.6209.pdf&embedded=true)

(Q. V. Le, M. A. Ranzato, R. Monga, M. Devin, K. Chen, G. S. Corrado, J. Dean and A. Y. Ng; 2012)


#### <a name="Mitosis Detection"></a>[Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks](http://www.idsia.ch/~juergen/miccai2013.pdf)

(D. C. Ciresan, A. Giusti, L. M. Gambardella and J. Schmidhuber; 2013)

Winner of the MICCAI 2013 Grand Challenge on Mitosis Detection


#### <a name="Hand Gesture Recognition"></a>[Max-Pooling Convolutional Neural Networks for Vision-based Hand Gesture Recognition](http://www.idsia.ch/~juergen/icsipa2011.pdf)

(J. Nagi, F. Ducatelle, G. A. Di Caro, D. Ciresan, U. Meier, A. Giusti, F. Nagi, J. Schmidhuber and L. M. Gambardella; Nov 2011)


 <br />

### <a name="Title index"></a>Index of titles:

[3D ShapeNets: A Deep Representation for Volumetric Shape Modeling](#3D ShapeNets) <br />
[Adversarial Feature Learning](#AFL) <br />
[Adversarially Learned Inference](#ALI) <br />
[Aligning 3D Models to RGB-D Images of Cluttered Scenes](#Aligning 3D Models to RGB-D Images of Cluttered Scenes) <br />
[A MultiPath Network for Object Detection](#A MultiPath Network for Object Detection) <br />
[A New Method to Visualize Deep Neural Networks](#Visualize DNNs) <br />
[A robust, coarse-to-ﬁne trafﬁc sign detection method](#Traffic Sign Detection) <br />
[Asynchronous Methods for Deep Reinforcement Learning](#Asynchronous Methods for Deep RL) <br />
[Building High-level Features Using Large Scale Unsupervised Learning](#High-level Features via Unsupervised Learning) <br />
[Continuous control with deep reinforcement learning](#DDPG) <br />
[Convolutional Neural Networks for joint object detection and pose estimation: A comparative study](#CNNs for object detection and pose estimation) <br />
[Cross Modal Distillation for Supervision Transfer](#RCNN-Depth-New) <br />
[Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition](#MNIST) <br />
[DeepBox: Learning Objectness with Convolutional Networks](#DeepBox) <br />
[DeepCamera: A Unified Framework for Recognizing Places-of-Interest based on Deep ConvNets](#DeepCamera) <br />
[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](#Deep Generative Image Models) <br />
[Deep Learning in Neural Networks: An Overview](#Deep Learning in Neural Networks: An Overview) <br />
[Deep Learning using Linear Support Vector Machines](#Deep Learning using Linear Support Vector Machines) <br />
[Deep multi-scale video prediction beyond mean square error](#Multi-scale video prediction) <br />
[Deep Residual Learning for Image Recognition](#Deep Residual Learning for Image Recognition) <br />
[Deep Sparse Rectifier Neural Networks](#Deep Sparse Rectifier Neural Networks) <br />
[DelugeNets: Deep Networks with Massive and Flexible Cross-layer Information Inflows](#DelugeNets) <br />
[Detection and Fine 3D Pose Estimation of Texture-less Objects in RGB-D Images](#Detection and 3D Pose Estimation) <br />
[Early Visual Concept Learning with Unsupervised Deep Learning](#Early Visual Concept Learning) <br />
[Exploring Person Context and Local Scene Context for Object Detection](#Context for ObjDet) <br />
[FaceNet: A Unified Embedding for Face Recognition and Clustering](#FaceNet) <br />
[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](#ELUs) <br />
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](#Faster R-CNN) <br />
[Fast Predictive Image Registration](#Fast Predictive Image Registration) <br />
[Fast R-CNN](#Fast R-CNN) <br />
[Fractional Max-Pooling](#Fractional Max-Pooling) <br />
[Freeze-Thaw Bayesian Optimization](#Freeze-Thaw Bayesian Optimization) <br />
[Going Deeper with Convolutions](#GoogLeNet) <br />
[Gradient-based Hyperparameter Optimization through Reversible Learning](#HyperparamOpt) <br />
[Grid Long Short-Term Memory](#Grid LSTM) <br />
[High-Dimensional Continuous Control Using Generalized Advantage Estimation](#Generalized Advantage Estimation) <br />
[Human-level control through deep reinforcement learning](#Atari2) <br />
[Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](#Hyperband) <br />
[Identity Mappings in Deep Residual Networks](#ResNet with pre-activation) <br />
[ImageNet Classification with Deep Convolutional Neural Networks](#AlexNet) <br />
[Image-to-Image Translation with Conditional Adversarial Networks](#pix2pix) <br />
[Improving generalization performance using double backpropagation](#Double Backprop) <br />
[Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks](#Inside-Outside Net) <br />
[Invertible Conditional GANs for image editing](#IcGAN) <br />
[Large Scale Distributed Deep Networks](#Large Scale Distributed Deep Networks) <br />
[Learning a Predictable and Generative Vector Representation for Objects](#Vector Representation for Objects) <br />
[Learning both Weights and Connections for Efficient Neural Networks](#Weights and Connections Learning) <br />
[Learning Compound Multi-Step Controllers under Unknown Dynamics](#Learning Compound Multi-Step Controllers under Unknown Dynamics) <br />
[Learning Deconvolution Network for Semantic Segmentation](#Deconv) <br />
[Learning Descriptors for Object Recognition and 3D Pose Estimation](#Learning Descriptors for Object Recognition and 3D Pose Estimation) <br />
[Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection](#Hand-Eye Coordination via DL) <br />
[Learning Rich Features from RGB-D Images for Object Detection and Segmentation](#RCNN-Depth) <br />
[Learning to Segment Object Candidates](#Learning to Segment Object Candidates) <br />
[Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders](#Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders) <br />
[Look and Think Twice: Capturing Top-Down Visual Attention with Feedback Convolutional Neural Network](#Look Twice)
[Max-Pooling Convolutional Neural Networks for Vision-based Hand Gesture Recognition](#Hand Gesture Recognition) <br />
[“Memory foam” approach to unsupervised learning](#Memory foam) <br />
[Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks](#Mitosis Detection) <br />
[Multi-column deep neural network for traffic sign classification](#Traffic Sign Classification) <br />
[Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](#Street View House Numbers) <br />
[Network In Network](#NiN) <br />
[NIPS 2016 Tutorial: Generative Adversarial Networks](#GAN Tutorial) <br />
[On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models](#Learning to think) <br />
[On the importance of initialization and momentum in deep learning](#Initialization and Momentum) <br />
[Particular object retrieval with integral max-pooling of CNN activations](#Particular object retrieval with integral max-pooling of CNN activations) <br />
[Playing Atari with Deep Reinforcement Learning](#Atari1) <br />
[Real-Time Full-Body Human Attribute Classification in RGB-D Using a Tessellation Boosting Approach](#RGB-D Human Attribute Classification) <br />
[R-CNN minus R](#R-CNN minus R) <br />
[Recurrent Convolutional Neural Network for Object Recognition](#Recurrent CNNs for object recognition) <br />
[Recurrent Convolutional Neural Networks for Scene Labeling](#Recurrent CNNs for scene labeling) <br />
[Recurrent Spatial Transformer Networks](#Recurrent Spatial Transformer Networks) <br />
[ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](#ReNet) <br />
[Rethinking the Inception Architecture for Computer Vision](#Inception2) <br />
[R-FCN: Object Detection via Region-based Fully Convolutional Networks](#R-FCN) <br />
[RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features](#RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features) <br />
[Rich feature hierarchies for accurate object detection and semantic segmentation](#R-CNN) <br />
[Scene Labeling with LSTM Recurrent Neural Networks](#Scene Labeling with 2D LSTM) <br />
[Semantic Object Parsing with Graph LSTM](#Semantic Object Parsing) <br />
[SimTrack: A Simulation-based Framework for Scalable Real-time Object Pose Detection and Tracking](#SimTrack) <br />
[Sparse 3D convolutional neural networks](#Sparse 3D CNNs) <br />
[Spatially-sparse convolutional neural networks](#Spatially-sparse ConvNets) <br />
[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](#SPPNet) <br />
[Spatial Transformer Networks](#Spatial Transformer Networks) <br />
[Speed learning on the fly](#SpeedLearning) <br />
[Stacked What-Where Auto-encoders](#Stacked What-Where Auto-encoders) <br />
[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](#StackGAN) <br />
[Systematic evaluation of CNN advances on the ImageNet](#CNN advances) <br />
[The Predictron: End-To-End Learning and Planning](#The Pedictron) <br />
[Train faster, generalize better: Stability of stochastic gradient descent](#Stability of SGD) <br />
[Understanding the Bias-Variance Tradeoff](#Bias-Variance Tradeoff) <br />
[Understanding the difﬁculty of training deep feedforward neural networks](#Difficulty of Training DNN) <br />
[Unrolled Generative Adversarial Networks](#Unrolled GANs) <br />
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](#DCGAN) <br />
[Very Deep Convolutional Networks for Large-Scale Image Recognition](#VGG) <br />
[Visualizing and Understanding Convolutional Networks](#ZFNet) <br />
