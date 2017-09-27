from setuptools import setup

setup(name='qv_summary',
      version='0.1',
      description='This shows how to use or pretrained model of relevance model to score frames based on a given text query',
      author='Arun Balajee Vasudevan, ETH Zurich',
      author_email='arunv@vision.ee.ethz.ch',
      license='BSD',
      packages=['qvsumm'],
      include_package_data=True,
      install_requires=[
          'numpy','moviepy','theano','lasagne','scikit-image','pafy','gensim==0.12.3','youtube-dl'],
      zip_safe=False)
