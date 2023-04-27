const title = 'Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction'
const authors = [
	{'name' : 'David M. Klee', link : 'https://dmklee.github.io', profile: 'https://dmklee.github.io/assets/images/profile.jpg'},
   	{'name' : 'Ondrej Biza', link : 'https://sites.google.com/view/obiza', profile: 'https://www2.ccs.neu.edu/research/helpinghands/author/ondrej-biza/avatar_hu254aef77619c527a447e931c40730b0b_11873_270x270_fill_q100_lanczos_center.jpg'},
	{'name' : 'Robert Platt', link : 'https://www.khoury.northeastern.edu/people/robert-platt/', profile: 'https://www2.ccs.neu.edu/research/helpinghands/author/robert-platt/avatar_hu2f90b12bbbd594d46b2f07527bff6f72_38994_270x270_fill_q100_lanczos_center.jpg'},
   	{'name' : 'Robin Walters', link : 'https://www.khoury.northeastern.edu/people/robin-walters/', profile: 'https://pointw.github.io/extrinsic_page/img/robin.jpeg'},
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

// quick links
quick_links = body.append('div')
				  .style('display', 'flex')
				  .style('justify-content', 'center')
				  .style('margin', '0 auto')
				  .style('font-size', '0.8rem')
				  .style('font-style', 'italic')
				  .style('margin-top', '-10px')
				  
quick_links.append('a').text('code').style('padding', '0 10px').style('text-decoration', 'none')
		   .attr('href', 'https://github.com/dmklee/image2sphere')
quick_links.append('a').text('paper').style('padding', '0 10px').style('text-decoration', 'none')
		   .attr('href', 'https://arxiv.org/pdf/2302.13926')
quick_links.append('a').text('bibtex').style('padding', '0 10px').style('text-decoration', 'none')
		   .attr('href', './bibtex.txt')


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
idea_div = body.append('div').style('font-size', '1.0rem').style('text-align', 'justify')
idea_div.append('p').text(
`This work studies the pose prediction problem: given a single image of an object, predicts the 3D rotation of the object
relative to a canonical orientation. The pose prediction problem exhibits 3D rotation symmetry: when the object is rotated, 
its pose rotates by the same amount (shown below). 
`
)
idea_div.append('img').style('display', 'block')
		.style('margin', '0 auto')
		.attr('src', 'assets/rotating_mug.gif')
		.attr('width', '400px')

idea_div.append('p').text(`
Ideally, we want a pose prediction network that learns an equivariant mapping (e.g. a rotated
copy of the input produces a rotated copy of the output).  We can construct equivariant neural
networks using trainable group convolution operations.  2D convolution is a well known instance of group
convolution that is equivariant to shifts in the 2D plane.  For the pose prediction problem,
we are interested in SO(3) group convolutions (spherical convolution), which is pictured on the right.
`)

idea_div.append('img').style('display', 'block')
		.style('margin', '0 auto')
		.style('margin-top', '-20px')
		.attr('src', 'assets/conv_text.gif')
		.attr('width', '400px')

idea_div.append('p').text(`
Unfortunately, group convolution can only be applied to inputs that are transformable by the group, so it is not
possible to apply SO(3)-equivariant layers directly on the image input.  Instead, we propose a hybrid-equivariant
network, where non-equivariant layers are used to learn features that can be further processed with equivariant layers.
More concretely, we use a standard convolutional network to extract a feature map, then map features onto the 2-sphere
using an orthographic projection (shown below).  Once the features live on the 2-sphere (an SO(3) transformable space),
we perform SO(3) group convolutions to improve the networks generalization capabilities.
`)

idea_div.append('img').style('display', 'block')
		.style('margin', '0 auto')
		.attr('src', 'assets/proj-diagram.png')
		.attr('width', '400px')

idea_div.append('p').text(`
The SO(3) group convolution operations are performed efficiently in the Fourier domain.  Signals over the 2-sphere or
SO(3) can be mapped to the Fourier domain using the Fast Fourier Transform, where they are represented by coefficients of
spherical harmonics (visualized below) or Wigner-D matrices, respectively. 
`)

idea_div.append('img').style('display', 'block')
		.style('margin', '0 auto')
		.attr('src', 'assets/trunc_fourier.png')
		.style('transform', 'rotate(-90deg)')
		.attr('width', '300px')

idea_div.append('p').text(`
We propose a novel approach to modeling distributions over SO(3) compactly by parametrizing in the Fourier domain.  Our approach
is simple, yet highly expressive.  Here we show examples of distributions our method generates for the SYMSOL dataset; it can accurately
model complex distributions resulting from objects with discrete or continuous symmetries.
`)

idea_div.append('img')
		.style('margin', 'auto 0')
		.attr('src', 'assets/symsol_pred.png')
		.attr('width', '800px')

idea_div.append('p').text(`
We also show state-of-the-art performance on the challening PASCAL3D+ dataset, which includes real images of objects from 12 classes. Example
predictions are shown below, which demonstrate how our model can naturally capture uncertainty.  The SO(3) equivariant layers of our method
improve sample efficiency, which is especially important for this dataset since it is difficult to capture all possible rotations of objects in the
dataset.
`)

idea_div.append('img').style('display', 'block')
		.style('margin', '0 auto')
		.attr('src', 'assets/pascal_pred.png')
		.attr('width', '800px')



make_header('Paper')
paper_div = body.append('div').style('line-height', 1.25).style('font-size', '1.0rem')
paper_div.append('div').text(
	'Published at the Eleventh International Conference on Learning Representations (ICLR 2023)'
)
paper_div.append('div').style('font-weight', 'bold').text("Notable-Top-5%")
paper_div.append('div').style('font-style', 'normal').append('a').attr('href', 'https://openreview.net/forum?id=_2bDpAtr7PI').style('text-decoration', 'none').text('[OpenReview]')
		 .append('span').append('a').attr('href', 'https://arxiv.org/pdf/2302.13926').style('text-decoration', 'none').text('[arXiv]')
authors_div = paper_div.append('div').style('font-weight', 'normal').style('padding', '20px 0 12px 0').style('display', 'flex').style('justify-content', 'flex-start')

for (let i=0; i < authors.length; i++) {
	let author = authors_div.append('div').style('display', 'grid').style('padding-right', '10px')
	author.append('img')
		  .attr('src', authors[i]['profile'])
		  .attr('width', '80px')
		  .style('border-radius', '50%')
		  .style('margin', '0 auto')
		  .style('padding', '2px')
		  
	author.append('a')
		  .attr('href', authors[i]['link']).style('text-decoration', 'none')
		  .text(authors[i]['name'])
		  .style('padding-right', '8px')
}
paper_div.append('div').style('font-weight', 'normal').text(
	'Khoury College of Computer Sciences'
)
paper_div.append('div').style('font-weight', 'normal').text(
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

