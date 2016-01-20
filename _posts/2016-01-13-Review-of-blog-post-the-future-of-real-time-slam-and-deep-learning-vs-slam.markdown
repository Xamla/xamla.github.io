---
layout: post
group: review
title: "Review of blog post: The Future of Real-Time SLAM and Deep Learning vs SLAM"
date:   2016-01-13 13:51:00
categories: update
---

**Blog post:**
[The Future of Real-Time SLAM and Deep Learning vs SLAM](http://www.computervisionblog.com/2016/01/why-slam-matters-future-of-real-time.html)
(Tomasz Malisiewicz; 13 Jan 2016)

**Description:** <br />
This blog post consists of three parts: 
* Part I: A brief introduction to SLAM
* Part II: A summary of T. Malisiewicz' workshop ["Future of Real-Time SLAM"](http://wp.doc.ic.ac.uk/thefutureofslam/programme/) at the [ICCV 2015](http://pamitc.org/iccv15/).
* Part III: Some conclusions from the deep learning-focused panel discussion at the end of the session (*"Deep learning vs SLAM"*).

# Part I: A brief introduction to SLAM
SLAM (Simultaneous Localization and Mapping) enables to simultaneously build 3D maps of the world while tracking the location and orientation of the camera. It focuses on geometric problems (i.e. enabling a robot to go towards a fridge without hitting a wall), while Deep Learning focuses on recognition problems (i.e. identifying items inside a fridge).
Moreover, SLAM is a real-time version of Structure from Motion (SfM) and one of its most important applications are self-driving cars.

# Part II: Workshop summary
* ["15 years of vision-based SLAM"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/slides_ajd.pdf) by Andrew Davison (Imperial College London) <br \>
   Historical overview of SLAM. Only very little has changed in SLAM during the last years, i.e. today's SLAM systems do not look much different than they did 8 years ago. Some of the most successful and memorable systems are: MonoSLAM, PTAM, FAB-MAP, DTAM and KinectFusion.
* ["Dense Continuous-Time Tracking and Mapping with Rolling Shutter RGB-D Cameras"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/kerl_etal_iccv2015_futureofslam_talk.pdf) by Christian Kerl (TU Munich)  <br \>
   Presentation of a dense tracking method to estimate a continuous-time trajectory, while usually most SLAM systems estimate camera poses at a discrete number of time steps. A big part of the talk was focused on undoing the damage of rolling shutter cameras.
* ["Semi-Dense Direct SLAM"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/ICCV-SLAM-Workshop_JakobEngel.pdf) by Jakob Engel (TU Munich)  <br \>
   Presentation of Engel's **LSD-SLAM** (Large-Scale Direct Monocular SLAM) system, which does not use corners or any local features (in contrast to many other SLAM systems). Direct tracking is performed by image-to-image alignment using a coarse-to-fine algorithm with a robust Huber loss. Rather than relying on image features, the algorithms is effectively performing “texture tracking”. Global mapping is performed by creating and solving a pose graph "bundle adjustment" optimization problem, and all of this works in real-time. The method is semi-dense because it only estimates depth at pixels solely near image boundaries. LSD-SLAM output is denser than traditional features, but not fully dense like Kinect-style RGBD SLAM. After this overview of the original LSD-SLAM system, some extensions to the initial system, were presented, namely **"Omni LSD-SLAM"**, which in contrast to the standard pinhole model, allows for a large field of view (ideally more than 180°), and **"Stereo LSD-SLAM"**, which extends LSD-SLAM to a binocular camera rig. It optimizes a pose graph in SE(3) and moreover includes a correction for auto exposure (*automatische Belichtung*), since from Engel's talk outliers are often caused by over-exposed image pixels and tend to be a problem. The goal of auto-exposure correcting is to make the error function invariant to affine lighting changes. The underlying parameters of the color-space affine transform are estimated during matching, but thrown away to estimate the image-to-image error. In addition, Engel showed a little bit of his current research about integrating both stereo and inertial sensors and finally he presented the differences between feature-based and direct SLAM methods.
* ["The challenges of Large-Scale Localization and Mapping"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/Sattler_challenges_large_scale_loc_and_mapping.pdf) by Torsten Sattler (ETH Zurich) <br \>
   ...
* ["Should we still do sparse-feature based SLAM?"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/ICCV15_SLAMWS_RaulMur.pdf) by Raúl Mur-Artal (University of Zaragoza) <br \>
   Feature-based vs Direct-Methods
* ["Google Project Tango: SLAM and dense mapping for end-users"]() by Simon Lynen (Google Zurich) <br \>
   Project Tango and Visual loop-closure for image-2-image constraints
* ["ElasticFusion: real-time dense SLAM without a pose graph"](http://wp.doc.ic.ac.uk/thefutureofslam/wp-content/uploads/sites/93/2015/12/ElasticFusion.pdf) by Stefan Leutenegger (Imperial College)  <br \>
   ElasticFusion is DenseSLAM without a pose-graph ...
* ["Dense SLAM in dynamic scenes"]() by Richard Newcombe () <br \>
   DynamicFusion... 

# Part III: Deep Learning vs SLAM

... to be continued.
