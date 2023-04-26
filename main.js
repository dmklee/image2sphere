const title = 'Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction'
const authors = [
	{'name' : 'David M. Klee', link : 'https://dmklee.github.io'},
   	{'name' : 'Ondrej Biza', link : 'https://sites.google.com/view/obiza'},
	{'name' : 'Robert Platt', link : 'https://www.khoury.northeastern.edu/people/robert-platt/'},
   	{'name' : 'Robin Walters', link : 'https://www.khoury.northeastern.edu/people/robin-walters/'},
]
const associations = [
	{'name' : 'Khoury College at Northeastern University',
	 'link' : 'https://www.khoury.northeastern.edu/',
	 'logo' : 'assets/khoury_logo.png',
	},
]
const colabs = [
	{'name' : 'Colab: Intro. to Spherical Harmonics',
	 'link' : 'https://colab.research.google.com/gist/dmklee/a05c24e0b3f5a36dc9ab6765ce2f97aa/introtosphericalharmonics.ipynb',
	 'logo' : 'assets/sphharm.png',
	},
	{'name' : 'Colab: Visualize Predictions',
	 'link' : 'https://colab.research.google.com/github/dmklee/image2sphere/blob/main/visualize_trained_predictions.ipynb',
	 'logo' : 'assets/google_pred.png',
	},
	{'name' : 'Colab: I2S Model Walkthrough',
	 'link' : 'https://colab.research.google.com/github/dmklee/image2sphere/blob/main/model_walkthrough.ipynb',
	 'logo' : 'assets/model_walkthrough.png',
	},
]

const abstract_text = 'Predicting the pose of objects from a single image is an important but difficult computer vision problem. Methods that predict a single point estimate do not predict the pose of objects with symmetries well and cannot represent uncertainty.  Alternatively, some works predict a distribution over orientations in SO(3). However, training such models can be computation- and sample-inefficient. Instead, we propose a novel mapping of features from the image domain to the 3D rotation manifold. Our method then leverages SO(3) equivariant layers, which are more sample efficient, and outputs a distribution over rotations that can be sampled at arbitrary resolution. We demonstrate the effectiveness of our method at object orientation prediction, and achieve state-of-the-art performance on the popular PASCAL3D+ dataset. Moreover, we show that our method can model complex object symmetries, without any modifications to the parameters or loss function.' 

function make_header(name) {
	body.append('div')
		.style('margin', '30px 0 10px 0')
		.style('padding-left', '4px')
		.style('padding-bottom', '4px')
		.style('border-bottom', '2px #999999 solid')
		.style('width', '100%')
		.append('p')
		.style('font-size', '1.3rem')
		.style('font-weight', 'bold')
		//.style('font-style', 'italic')
		.style('color', '#5e5e5e')
		.style('margin', '2px 4px')
		.text(name)
	
}

const max_width = '800px';

var body = d3.select('body')
			 .style('max-width', max_width)
			 .style('margin', '30px auto')
			 .style('margin-top', '80px')
			 .style("font-family", "Georgia")
			 .style("font-size", "1.1rem")

// title
body.append('p')
	.style('font-size', '2.2rem')
	.style('font-weight', 300)
	.style('text-align', 'center')
	.style('margin', '20px auto')
	.text(title)

// authors
//var authors_div = body.append('div').attr('class', 'flex-row').style('font-size', '0.8rem')
//for (let i=0; i < authors.length; i++) {
	//authors_div.append('a')
				//.attr('href', authors[i]['link'])
				//.text(authors[i]['name'])
				//.style('margin', '10px')
//}

// associations
//var associations_div = body.append('div').attr('class', 'flex-row')
//for (let i=0; i < associations.length; i++) {
	//associations_div.append('a')
					//.attr('href', associations[i]['link'])
					//.append('img')
					//.attr('src', associations[i]['logo'])
					//.style('height', '50px')
//}


// main figure
var fig_div = body.append('div')
	.attr('class', 'flex-row')
fig_div
	.append('img')
	.style('margin', 'auto 0')
	.attr('src', 'assets/figure1.png')
	.attr('width', '800px')


// abstract
body.append('div')
	.style('width', '80%')
	.style('margin', '10px auto')
	.style('text-align', 'justify')
	.style('line-height', 1.3)
	.style('font-size', '1rem')
	.append('span').style('font-weight', 'bold').text('Abstract: ')
	.append('span').style('font-weight', 'normal')
	.text(abstract_text)

make_header('Idea')
idea_div = body.append('div')//.style('display', 'flex')
idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/rotating_mug.gif')
		.attr('width', '400px')

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/conv_text.gif')
		.attr('width', '400px')

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/proj-diagram.png')
		.attr('width', '400px')

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/trunc_fourier.png')
		.attr('width', '400px')

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/pascal_pred.png')
		.attr('width', '800px')

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/symsol_pred.png')
		.attr('width', '800px')


make_header('Paper')
paper_div = body.append('div').style('line-height', 1.25).style('font-size', '1.0rem')
paper_div.append('div').text(
	'Published at the Eleventh International Conference on Learning Representations (ICLR 2023)'
)
paper_div.append('div').style('font-weight', 'bold').text("Notable-Top-5%")
paper_div.append('div').style('font-style', 'normal').append('a').attr('href', 'https://openreview.net/forum?id=_2bDpAtr7PI').style('text-decoration', 'none').text('[OpenReview]')
		 .append('span').append('a').attr('href', 'https://arxiv.org/pdf/2302.13926').style('text-decoration', 'none').text('[arXiv]')
authors_div = paper_div.append('div').style('font-weight', 'normal')

for (let i=0; i < authors.length; i++) {
	authors_div.append('a')
				.attr('href', authors[i]['link']).style('text-decoration', 'none')
				.text(authors[i]['name'])
				.style('padding-right', '8px')
}
authors_div = paper_div.append('div').style('font-weight', 'normal').text(
	'Khoury College of Computer Sciences'
)
authors_div = paper_div.append('div').style('font-weight', 'normal').text(
	'Northeastern University'
)
	

make_header('Video')
body.append('div')
	.append('iframe')
	.attr('title', 'I2S Presentation')
	.attr('width', '800')
	.attr('height', '450')
	.attr('type', "text/html")
	.attr('frameborder', '0')
	.attr('src', 'https://www.youtube.com/embed/yZz3umtyEEk')

make_header('Code')
code_body = body.append('div').style('font-size', '1.0rem')
code_body.append('span')
	.text('The code is available ')
code_body.append('a')
	.attr('href', 'https://github.com/dmklee/image2sphere')
	.text('here')
code_body.append('span')
	.text(`. We provide training scripts to reproduce results, pre-trained model weights. 
For a quick intro to the code, check out the Colab notebooks:
`)

for (let i=0; i < colabs.length; i++) {
	colab_div = body.append('div')
		.attr('class', 'flex-row')
		.style('max-height', '90px')
		.style('border', '1px solid black')
		.style('border-radius', '10px')
		.style('overflow', 'hidden')
		.style('position', 'relative')
		.on('mouseover', function(d) {
			d3.select(this).style('border', '3px solid black')
		})
		.on('mouseout', function(d) {
			d3.select(this).style('border', '1px solid black')
		})
	colab_div.append('a')
		.attr('href', colabs[i].link)
		.append('img')
		.attr('src', colabs[i].logo)
		.style('width', '100%')
		.style('transform', 'translateY(-20%)')
	colab_div.append('p').style('font-weight', 'bold')
		  .style('position', 'absolute')
		  .style('left', '2%')
		  .style('top', '-5%')
		  .text(colabs[i].name)
}


make_header('Citation')
body.append('div')
	.append('p')
	.style('border-radius', '6px')
	.style('padding', '10px')
	.style('background-color', '#eee')
	.append('pre')
	.style('font-size', '0.8rem')
	.style('line-height', '1.6')
	.text(`@inproceedings{
  klee2023image2sphere,
  title={Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction},
  author={David M. Klee and Ondrej Biza and Robert Platt and Robin Walters},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://arxiv.org/pdf/2302.13926}
}`)

make_header('Contact')
code_body = body.append('div').style('font-size', '0.8rem')
code_body.append('span')
	.text('For questions, please contact David Klee at: klee [dot] d [at] northeastern [dot] edu')


// common syntax
body.selectAll('.flex-row')
	.style('margin', '20px auto')
    .style('display', 'flex')
    .style('justify-content', 'center')
    .style('flex-direction', 'row')
    .style('width', '100%')

body.selectAll('a').style('color', 'blue')
body.selectAll('.content')
	.style('margin', '20px auto')
	.style('width', '90%')

