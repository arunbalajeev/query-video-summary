## Query-adaptive Video Summarization via Quality-aware Relevance Estimation

Code for the paper:
"Query-adaptive Video Summarization via Quality-aware Relevance Estimation"- ACM Multimedia 2017
Arun Balajee Vasudevan\*, Michael Gygli\*, Anna Volokitin, Luc Van Gool(\* denotes equal contribution)  
CVLab, ETH Zurich

### Installation

1. Download this repository or clone with Git, and then `cd` into the root directory of the repository.
2. Install the requirements with `pip install -r requirements.txt`
3. Run `python setup.py install --user` to install the package __qvsumm__.

This will suffice to run the notebooks below. For the package to work from any 
location, additionally run `export QVSUM_DATA_DIR=DATA_DIR`,
where `DATA_DIR` is the absolute path of the directory `./query-video-summary/data`.
This is necessary so that the model files can be found.
How to download the models is described in the notebooks.

Note: We use Lasagne for our implementation. The code is tested for cuDNN version==3.0.

### Getting Started

1. Thumbnail Extraction - This demo shows how to extract query relevant thumbnails from a video after scoring all the video frames based on its relevance to the text query. It takes inputs- text query and video url.  
Run `thumbnail_demo.ipynb`.

2. Summarization- This demo shows how to get the query relevant summary of the video as a set of keyframes. It takes inputs- text query and video url.  
Run `summarization_demo.ipynb`.

![Image](https://people.ee.ethz.ch/~arunv/images/summarize_results.png)

If you use the relevance prediction of this code please cite:
    
    Arun Balajee Vasudevan*, Michael Gygli*, Anna Volokitin, Luc Van Gool
    "Query-adaptive Video Summarization via Quality-aware Relevance Estimation"
    ACM Multimedia 2017
    (* denotes equal contribution)  

If you use the summarization code, please also cite the following paper, 
which provides code for maximizing submodular mixtures:

    Michael Gygli, Helmut Grabner, Luc Van Gool
    "Video Summarization by Learning Submodular Mixtures of Objectives,"
    IEEE CVPR 2015


