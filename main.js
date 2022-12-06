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
const abstract_text = 'Predicting the pose of objects from a single image is an important but difficult computer vision problem. Methods that predict a single point estimate do not predict the pose of objects with symmetries well and cannot represent uncertainty.  Alternatively, some works predict a distribution over orientations in SO(3). However, training such models can be computation- and sample-inefficient. Instead, we propose a novel mapping of features from the image domain to the 3D rotation manifold. Our method then leverages SO(3) equivariant layers, which are more sample efficient, and outputs a distribution over rotations that can be sampled at arbitrary resolution. We demonstrate the effectiveness of our method at object orientation prediction, and achieve state-of-the-art performance on the popular PASCAL3D+ dataset. Moreover, we show that our method can model complex object symmetries, without any modifications to the parameters or loss function.' 

function make_header(name) {
	body.append('div')
		.style('margin', '30px 0 10px 0')
		.style('padding-left', '8px')
		.style('padding-bottom', '4px')
		.style('border-bottom', '1px #555 solid')
		.style('width', '100%')
		.append('p')
		.style('font-size', '1.5rem')
		.style('font-style', 'italic')
		.style('margin', '2px 4px')
		.text(name)
}

const max_width = '800px';

var body = d3.select('body')
			 .style('max-width', max_width)
			 .style('margin', '60px auto')
			 .style('margin-top', '100px')
			 .style("font-family", "Garamond")
			 .style("font-size", "1.2rem")

// title
body.append('p')
	.style('font-size', '1.8rem')
	.style('font-weight', 500)
	.style('text-align', 'center')
	.style('margin', '20px auto')
	.text(title)

// authors
var authors_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < authors.length; i++) {
	authors_div.append('a')
				.attr('href', authors[i]['link'])
				.text(authors[i]['name'])
				.style('margin', '10px')
}

// associations
var associations_div = body.append('div').attr('class', 'flex-row')
for (let i=0; i < associations.length; i++) {
	associations_div.append('a')
					.attr('href', associations[i]['link'])
					.append('img')
					.attr('src', associations[i]['logo'])
					.style('height', '70px')
}


// main figure
//var fig_div = body.append('div')
	//.attr('class', 'flex-row')
//fig_div
	//.append('img')
	//.style('margin', 'auto 0')
	//.attr('src', 'assets/figure1.png')
	//.attr('width', '600px')


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

make_header('Paper')
body.append('div').style('line-height', 1.4).style('font-weight', 'bold').style('font-size', '0.9rem').text(title)
	.append('div').style('font-weight', 'normal').text(authors.map(d => ' '+d.name))
	.append('div').style('font-style', 'italic').text("Preprint")
	.append('div').style('font-style', 'normal').append('a').attr('href', 'assets/paper.pdf').text('[PDF]')
	

make_header('Code')
body.append('div')
	.text('View the code on Github ')
	.append('a')
	.attr('href', 'https://github.com/dmklee/image2sphere')
	.text('here.')

//make_header('Citation')
//body.append('div')
	//.append('p')
	//.style('border-radius', '6px')
	//.style('padding', '10px')
	//.style('background-color', '#eee')
	//.append('pre')
	//.style('font-size', '0.8rem')
	//.style('line-height', '1.6')
	//.text(`@misc{imagetoico2022,
  //title = {I2I: Image to Icosahedral Projection for $\mathrm{SO}(3)$ Object Reasoning from Single-View Images},
  //author = {Klee, David and Biza, Ondrej and Platt, Robert and Walters, Robin},
  //journal = {arXiv preprint arXiv:2207.08925},
  //year = {2022},
//}`)

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

