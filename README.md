# video-enhanced-classification

Video streams contain more information than individual images. They contain time series in formation across the image frames and they contain variations in 
lighting and camera motion, for example. When a short video stream focuses on only one scene in a video shot of several seconds, this time, lighting and 
attention focus information can be used to create a series of images that are semantically identical, i.e. depicting the exact same plants in the wild, 
across slightly varying angles and lighting conditions. 

Moreover, these images contain indirect information on scene significance as videographers (as photographers) focus the center of the view field on specific 
items of interest. Our video enhanced classification pipeline takes all of these cues into account to improve the output quality of a classification effort.

Together these operations produce a set of varied image sections of constant semantics. This combination of variety of image details and constancy in image c
ontent is one key element to using the image data to improve classifier performance.

We can not know a priori which results of the individual classifiers are correct. We address this problem by combining the outputs of all the classifiers
to increase confidence in the respective results. Shared results across the classifiers are considered more likely to be correct, results that occur in o
nly one classifier, on the other hand, are considered more likely to be spurious.

We tally all the responses (across all test images and classifier architectures) and rank the results by frequency of detected category 
(at a selected probability threshold). From this ranked list we pick the top items as the most likely results. Below is an example of the output of 
this process on a 19 second field video depicting cacao and banana in a forest full of rich Baliense plant life visible to the camera:

TALLY:  {'sugarpalm': 6, 'cacao': 4, 'taro': 1, 'banana': 3, 'bamboo': 1, 'dragonfruit': 1}
 > FINAL TALLY above single instance  ['sugarpalm', 'cacao', 'banana']

The system correctly detects the presence of cacao and banana, but also believes that sugarpalm is dominantly represented (it is not). 

