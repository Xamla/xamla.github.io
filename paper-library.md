---
layout: page
title: Paper Library
permalink: /paper-library/
mathjax: true
---

#### [\\(\rightarrow\\) Index of titles](#Title index) <br />


### Papers with description:


#### <a name="Deconv"></a>[Learning Deconvolution Network for Semantic Segmentation](http://cvlab.postech.ac.kr/research/deconvnet/)

(H. Noh,	S. Hong	and B. Han; 17 May 2015)

*...*


#### <a name="Learning to Segment Object Candidates"></a>[Learning to Segment Object Candidates](http://arxiv.org/abs/1506.06204)

(P. O. Pinheiro, R. Collobert and P. Dollar; 20 Jun 2015)

*ConvNet approach for generating object proposals for the object detection task. Main difference to "Faster R-CNN": The method presented here (called "DeepMask") generates segmentation proposals instead of less informative bounding box proposals. The core of this approach is a ConvNet which jointly predicts a segmentation mask given an input patch and assigns an object score corresponding to how likely the patch is to contain an object. A large part of the network is shared between those two tasks: only the last few network layers are specialized for separately outputting a mask and score prediction. The model is trained by optimizing a cost function that targets both tasks simultaneously. In detail, a VGG-A ConvNet architecture (initialized with pre-trained ImageNet classification parameters and consisting of 3x3 convolutions, ReLUs and 2x2 max pooling layers) is used, where the final fully connected layers and the last max-pooling layer are removed, because the spatial information provided in the convolutional feature maps is needed for inferring segmentation masks. The branch of the network dedicated to segmentation is composed of a single 1x1 convolution with ReLU and a classification layer, consisting of several pixel classifiers. The classification layer moreover is decomposed into two linear layers with no non-linearity in between (a low-rank variant of using fully connected linear classifiers, reducing the number of network parameters while allowing each pixel classifier to use information from the entire feature map). The branch of the network dedicated to scoring is composed of a 2x2 max-pooling and two fully connected layers with ReLUs and dropout. The loss function is a sum of binary logistic regression losses, one for each location of the segmenation network and one for the object score. An alternation between backpropagation through the segmenation and scoring branch is performed. Generalization capabilities of the model are demonstrated by testing it on object categories not seen during training. For this, segmentation training with only positive scored objects is critical, since this way the network attempts to generate a segmentation mask at every patch, even if no known object is present. During full image inference the model is densely applied at multiple locations and scales. Since all computations can be computed convolutionally, the full image inference procedure is still efficient. For the implementation of all experiments Torch7 has been used.*


#### <a name="Deep Generative Image Models"></a>[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](http://arxiv.org/abs/1506.05751)

(E. Denton, S. Chintala, A. Szlam and R. Fergus; 18 Jun 2015)


#### <a name="Inside-Outside Net"></a>[Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks](http://arxiv.org/abs/1512.04143)

(S. Bell, C. L. Zitnick, K. Bala and R. Girshick; 14 Dec 2015)


#### <a name="Deep Residual Learning for Image Recognition"></a>[Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)

(K. He, X. Zhang, S. Ren and J. Sun; 10 Dec 2015) 

*Gewinner der "ImageNet Large Scale Visual Recognition Challeng" (ILSVRC) 2015 in den Rubriken "Object Detection" und "Localization".*


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

[A robust, coarse-to-ﬁne trafﬁc sign detection method](#Traffic Sign Detection) <br />
[Building High-level Features Using Large Scale Unsupervised Learning](#High-level Features via Unsupervised Learning) <br />
[Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition](#MNIST) <br />
[DeepBox: Learning Objectness with Convolutional Networks](#DeepBox) <br />
[DeepCamera: A Unified Framework for Recognizing Places-of-Interest based on Deep ConvNets](#DeepCamera) <br />
[Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](#Deep Generative Image Models) <br />
[Deep Learning in Neural Networks: An Overview](#Deep Learning in Neural Networks: An Overview) <br />
[Deep Learning using Linear Support Vector Machines](#Deep Learning using Linear Support Vector Machines) <br />
[Deep multi-scale video prediction beyond mean square error](#Multi-scale video prediction) <br />
[Deep Residual Learning for Image Recognition](#Deep Residual Learning for Image Recognition) <br />
[Deep Sparse Rectifier Neural Networks](#Deep Sparse Rectifier Neural Networks) <br />
[Detection and Fine 3D Pose Estimation of Texture-less Objects in RGB-D Images](#Detection and 3D Pose Estimation) <br />
[FaceNet: A Unified Embedding for Face Recognition and Clustering](#FaceNet) <br />
[Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](#ELUs) <br />
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](#Faster R-CNN) <br />
[Fast R-CNN](#Fast R-CNN) <br />
[Fractional Max-Pooling](#Fractional Max-Pooling) <br />
[Freeze-Thaw Bayesian Optimization](#Freeze-Thaw Bayesian Optimization) <br />
[Going Deeper with Convolutions](#GoogLeNet) <br />
[Gradient-based Hyperparameter Optimization through Reversible Learning](#HyperparamOpt) <br />
[Grid Long Short-Term Memory](#Grid LSTM) <br />
[Human-level control through deep reinforcement learning](#Atari2) <br />
[ImageNet Classification with Deep Convolutional Neural Networks](#AlexNet) <br />
[Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks](#Inside-Outside Net) <br />
[Large Scale Distributed Deep Networks](#Large Scale Distributed Deep Networks) <br />
[Learning both Weights and Connections for Efficient Neural Networks](#Weights and Connections Learning) <br />
[Learning Compound Multi-Step Controllers under Unknown Dynamics](#Learning Compound Multi-Step Controllers under Unknown Dynamics) <br />
[Learning Deconvolution Network for Semantic Segmentation](#Deconv) <br />
[Learning Descriptors for Object Recognition and 3D Pose Estimation](#Learning Descriptors for Object Recognition and 3D Pose Estimation) <br />
[Learning to Segment Object Candidates](#Learning to Segment Object Candidates) <br />
[Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders](#Learning Visual Feature Spaces for Robotic Manipulation with Deep Spatial Autoencoders) <br />
[Look and Think Twice: Capturing Top-Down Visual Attention with Feedback Convolutional Neural Network](#Look Twice)
[Max-Pooling Convolutional Neural Networks for Vision-based Hand Gesture Recognition](#Hand Gesture Recognition) <br />
[“Memory foam” approach to unsupervised learning](#Memory foam) <br />
[Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks](#Mitosis Detection) <br />
[Multi-column deep neural network for traffic sign classification](#Traffic Sign Classification) <br />
[Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](#Street View House Numbers) <br />
[Network In Network](#NiN) <br />
[On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models](#Learning to think) <br />
[On the importance of initialization and momentum in deep learning](#Initialization and Momentum) <br />
[Playing Atari with Deep Reinforcement Learning](#Atari1) <br />
[Real-Time Full-Body Human Attribute Classification in RGB-D Using a Tessellation Boosting Approach](#RGB-D Human Attribute Classification) <br />
[R-CNN minus R](#R-CNN minus R) <br />
[Recurrent Spatial Transformer Networks](#Recurrent Spatial Transformer Networks) <br />
[ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](#ReNet) <br />
[Rethinking the Inception Architecture for Computer Vision](#Inception2) <br />
[RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features](#RGB-D Object Recognition and Pose Estimation based on Pre-trained Convolutional Neural Network Features) <br />
[Rich feature hierarchies for accurate object detection and semantic segmentation](#R-CNN) <br />
[SimTrack: A Simulation-based Framework for Scalable Real-time Object Pose Detection and Tracking](#SimTrack) <br />
[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](#SPPNet) <br />
[Spatial Transformer Networks](#Spatial Transformer Networks) <br />
[Speed learning on the fly](#SpeedLearning) <br />
[Understanding the Bias-Variance Tradeoff](#Bias-Variance Tradeoff) <br />
[Understanding the difﬁculty of training deep feedforward neural networks](#Difficulty of Training DNN) <br />
[Very Deep Convolutional Networks for Large-Scale Image Recognition](#VGG) <br />
[Visualizing and Understanding Convolutional Networks](#ZFNet) <br />

