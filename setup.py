from setuptools import setup, find_packages

setup(
	name='deepsvfilter',
	packages=find_packages(),
	version='0.1.0',
	scripts=['deepsvfilter/DeepSVFilter','deepsvfilter/vcf2bed','deepsvfilter/extract_typical_SVs'],
	
        install_requires=[
                'tensorflow-gpu==1.15.0',
		'matplotlib==3.1.0',
		'numpy<2.0,>=1.16.0',
                'opencv-python==3.1.0.4',
                'Pillow==8.3.2',
		'pysam==0.15.4',
                'scikit-learn==0.19.2',
                'scipy',
	],
        url='https://github.com/yongzhuang/DeepSVFilter',
	license='MIT',
	author='Yongzhuang Liu',
	author_email='yongzhuang.liu@hit.edu.cn',
	description='A Deep Learning Approach for Filtering Structural Variants in Short Read Sequencing Data',
	long_description=open('README.md').read(),
	long_description_content_type="text/markdown",
)
