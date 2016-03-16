---
layout: post
group: review
title:  "Review of Learning Hand-Eye Coordination via DL"
date:   2016-03-10 13:51:00
categories: update
mathjax: true
---

**Paper:**
[Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection](http://arxiv.org/abs/1603.02199)
(S. Levine, P. Pastor, A. Krizhevsky and D. Quillen; 7 Mar 2016)

**Description:** <br />
New learning-based approach to hand-eye coordination for robotic grasping from monocular images. 

**Method consists of two parts:** <br />
* A grasp success prediction network \\(g(I_t, v_t)\\), i.e. a deep convolutional neural network (CNN), which as input gets an image \\(I_t\\) and a task-space motion command \\(v_t\\) and as ouput returns the probability of motion command \\(v_t\\) resulting in a successful grasp. 
* A servoing function \\(f(I_t)\\), which uses the prediction network to continuously update the robot’s motor commands to servo the gripper to a success grasp.
By continuously choosing the best predicted path to a successful grasp, the servoing mechanism provides the robot with fast feedback to perturbations and object motion, as well as robustness to inaccurate actuation.

**Slight drawback:** <br />
Currently, **only vertical pinch grasps** are considered (though extensions to other grasp parameterizations would be straightforward).

**Important advantage:** <br />
The model **does not require** the **camera** to be precisely **calibrated** with respect to the end-effector, 
but instead continuously uses visual feedback to determine the spatial relationship between the gripper and graspable 
objects in the scene. 

**Training:** <br />
* Large dataset of over 800000 grasp attempts collected over the course of two months, using between 6 and 14 robotic manipulators at any given time.
* Slight differences in camera placement (always behind the robot) and slight differences in wear and tear on each robot resulting in differences in the shape of the gripper fingers. 
* Each grasp \\(i\\) consists of \\(T\\) time steps. At each time step \\(t\\), the robot records the current image \\(I_t^i\\) and the current pose \\(p_t^i\\), and then chooses a direction along which to move the gripper. At the final time step \\(T\\), the robot closes the gripper and evaluates the success of the grasp, producing a label \\(l_i\\). 
  The final dataset contains samples \\((I_t^i, p_T^i − p_t^i, l_i)\\) that consist of the image, a vector from the current pose to the final pose, and the grasp success label.
* The CNN is trained with a **cross-entropy loss** to match \\(l_i\\), causing the network to output the probability \\(p(l_i = 1)\\). 
  
The **servoing mechanism** uses the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp. 
Thereto a “small” optimization on \\(v_t\\) is performed using **three iterations of the cross-entropy method (CEM)**, a simple derivative-free optimization algorithm.
CEM samples a batch of N values at each iteration, fits a Gaussian distribution to M < N of these samples, and then samples a new batch of N from this Gaussian. 
Here: N = 64, M = 6

**Two heuristics for gripper and robot motion:** <br />
* The gripper is closed whenever the network predicts that no motion will succeed with a probability that is at least 90% 
  of the best inferred motion. <br />
  **Reason:** Stop the grasp early if closing the gripper is nearly as likely to produce a successful grasp as moving it.
* The gripper is raised off the table whenever the network predicts that no motion has a probability of success that is 
  less than 50% of the best inferred motion. <br />
  **Reason:** If closing the gripper now is substantially worse than moving it, the gripper is most likely not positioned 
  in a good configuration, and a large motion will be required. Therefore, raising the gripper off the table minimizes
  the chance of hitting other objects that are in the way. 
  
**Two methods of grasp success evaluation during data collection:** <br />
* The position reading on the gripper is greater than 1cm, indicating that the fingers have not closed fully 
  (only suitable for thick objects). 
* The images of the bin containing the objects recorded before and after a drop differ, indicating that there 
  has somenthing been in the gripper (“drop test”). 

**Experimental results:** <br />
* The presented method has been tested to be more robust to perturbations as movement of objects in the scene and 
  variability in actuation and gripper shape than an “open-loop approach” (without continuous feedback). <br />
  (“Open-loop approach”: 
    Scene is observed prior to the grasp, image patches are extracted, the patch with the highest 
    probability of a successful grasp is chosen, and then a known camera calibration is used to move the gripper to that 
    location.)
* Grasps automatically were adapted to the different material properties of the objects. 
* Even challenging (e.g. flat) objects could be grasped.
