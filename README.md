Frecog
==========

Frecog (pronounced like Free Cog), is a facial recognition project that utilizes
OpenCV to perform a Eigenfaces recogntion of faces. Currently, running the
application creates a GUI that shows your video feed, any detected faces and the
associated names.

Prerequisites
----------
 - Python2
 - numpy
 - cv2

Installation
----------
For each person to detect, add a new numbered folder in "data". Each identity
and corresponding data number should be added to the identities.txt file in as
shown in the files example, and each training image should be added to the
corresponding data number in training.txt.

To-Do
-----------
 - Merge the training.txt and identities.txt files.
 - Move variable storage to a configuration file.
 - Improve image processing for better recognition.
 - Add in eye detection(?).
