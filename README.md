# video informed image classification

![alt text](https://github.com/realtechsupport/video-enhanced-classification/blob/main/video-enhanced-CNN.png?raw=true)

Image classifiers work well on structured images, but they often fail on images with very high visual complexity captured in the wild in unusual configurations.

The project responds to the challenges encountered in the <a href="http://www.realtechsupport.org/new_works/return2bali.html">Return to Bali</a> project that seeks to apply machine learning to the field of Ethnobotany, the stud of how people interact with local flora. Even current state-of-the art-image classifiers such as <a href = "https://github.com/facebookresearch/detectron2"> Detectron2 </a> fail to robustly recognize plants from the dataset in the wild of the forests of Bali.

Video streams contain more information than individual images. They contain time series information across the image frames and they contain variations in 
lighting and camera motion, for example. When a short video stream focuses on only one scene in a video shot of several seconds, this time, lighting and 
attention focus information can be used to create a series of images that are semantically identical, i.e. depicting the same items
across slightly varying angles and lighting conditions. 

Moreover, these images contain indirect information on scene significance as videographers (as photographers) focus the center of the view field on specific 
items of interest. This video enhanced classification pipeline takes all of these cues into account to improve the output quality of a classification effort.

Together these operations produce a set of varied image sections of constant semantics. This combination of variety of image details and constancy in image content is one key element to using the image data to improve classifier performance.

We can not know a priori which results of the individual classifiers are correct. The approach addresses this problem by combining the outputs of all the classifiers to increase confidence in the respective results. Shared results across the classifiers are considered more likely to be correct, results that occur in only one classifier, on the other hand, are considered more likely to be spurious.

The code tallies all the responses (across all test images and classifier architectures) and ranks the results by frequency of detected category 
(at a selected probability threshold). From this ranked list the code picks the top items as the most likely results. 

Below is an example of the output of  this process on a 19 second field video depicting cacao and banana in a forest full of rich Balinese plant life visible to the camera:

TALLY:  {'sugarpalm': 6, 'cacao': 4, 'taro': 1, 'banana': 3, 'bamboo': 1, 'dragonfruit': 1}
 > FINAL TALLY above single instance  ['sugarpalm', 'cacao', 'banana']

The system correctly detects the presence of cacao and banana, but also believes that sugarpalm is dominantly represented (it is not). 
There is much room for improvement, but the approach opens the door to improved classification with compromised classifiers.

Here is a link to the trained detectron2 classifiers upon which the code depends:

https://u.pcloud.link/publink/show?code=kZItwBXZlME4a08Nw4HiIkpzAWGzBJ8b1fs7

Here is a link to the data sets used in the experiment:

https://u.pcloud.link/publink/show?code=kZJ6wBXZuSLYTcuNT0hl0Gldorg3PyHS0Dcy

