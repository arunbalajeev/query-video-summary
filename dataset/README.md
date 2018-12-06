### query_videoURLs.csv
This file contains queries and corresponding extracted videos (URLs) from YouTube.

### query_frame_annotations.csv
This file contains the URLs of the extracted video frames and the annotations of relevance and diversity for each frame from 5 different workers. Different annotations are uniquely identified by the assignment ID in the column 2.

## Dataset Annotation
We first annotate the video frames with query relevance labels, and then partition the frames into clusters according to visual similarity.
Relevance annotation ranges between 0 and 4 (Options for answers are “Trash”,“Not good”, “Good” and “Very Good”) for each frame.
Cluster annotations starts from 0 and ranges to an arbitary number. 0th cluster indicates Trash frames and are of low quality(e.g. blurred, bad contrast, etc.) while the cluster numbers >=1 indicates different groups. We obtain one clustering per worker, where each clustering consists of mutually exclusive subsets of video frames as clusters.


 
