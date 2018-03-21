# Neural-Style-Transfer
# Usage: 
	nst = NeuralStyleTransfer(contentImagePath,styleImagePath,modelPath)
		-- contentImagePath: the path to the image you want to transfer
		-- styleImagePath: the path to the image you want to get the style from
		-- modelPath: imagenet vgg-19 model in this repository
	nst.run(num_iterations = 300)
You can download the vgg-19 model in this link:https://mega.nz/#!pIFimCCA!9nFD0KJ_ysx0NWfEs90bPkjvBMUn1Y82pYF-FrWLHw8

after the Neural Style Transfer finishes its task, you will find a folder named output in the same directory, and the result will be there.

<div align="center">
	<img src="content.jpg" width="45%">
	<img src="style.jpg" width="45%">
</div>
<div align="center">
	<img src="output/generated_image.jpg" width="50%">
</div>

## What is Neural Style Transfer
Neural style transfer is you take a content image C and combine it with a style image S to get a generated image G such that image G is the content image drawn in the style of image S.

<div align="center">
	<img src="resources/Screen Shot 2018-03-20 at 11.09.42 PM.png" width="50%">
</div>

In order to understand the Neural style transfer, we need to look at the features extracted by ConvNet at various layers, both the shallow and the deeper layers of a ConvNet. If we have a ConvNet looks like the one below, and if we pick a unit in layer one and find nine image patches that maximize the unit's activation. We repeat the process for other units in layer one, then we can see that units in layer one have relatively simple features such as edge or a particular shade of color.

<img src="resources/Screen Shot 2018-03-20 at 11.24.58 PM.png" width="90%">
<img src="resources/Screen Shot 2018-03-20 at 11.25.21 PM.png" width="90%">

If we go deeper to the ConvNet, the deeper layers will see a larger region of the image where each pixel could hypothetically affect the output of these later layers of the neural network. We can see that layer 2 is looking for a bit more complex shapes and patterns than layer 1, and layer 3 is looking for rounded shapes and people. Layer 4 is detecting dogs, water etc, and layer 5 is detecting even more sophisticated things.

<div align="center">
<img src="/Users/jin/Desktop/Screen Shot 2018-03-20 at 11.38.51 PM.png" width="40%">
<img src="/Users/jin/Desktop/Screen Shot 2018-03-20 at 11.45.23 PM.png" width="40%">
</div>
<div align="center">
<img src="/Users/jin/Desktop/Screen Shot 2018-03-20 at 11.54.45 PM.png" width="40%">
<img src="/Users/jin/Desktop/Screen Shot 2018-03-20 at 11.54.56 PM.png" width="40%">
</div>

So we've gone a long way from detecting relatively simple things such as edges in layer 1 to textures in layer 2, up to detecting very complex objects in the deeper layers.

We now define a Cost function J(G) and use gradient descent to minimize the cost function to get our generated image G. 

<div align="center">
	<img src="resources/Screen Shot 2018-03-21 at 12.28.14 AM.png" width="70%">
</div>

The cost function has two parts, and the first part is a measure, or the cost, of how similar the content image compare to the generated image. Second part is the measure, or cost, of how similar is the style of image G compare to the style image S. Finally we will weight these with two hyperparameter alpha and beta to specify the weight of each cost.
The step to find generated image G is follows:

1. Initiate G randomly with dimension W x H x 3
2. Use gradient descent to minimize the cost function J(G)

Let calculate the content cost:
1. Choose a hidden layer l to compute the content cost. Usually we choose l not too shallow nor too deep in the neural network. If we choose a layer l that is too shallow then the generated image will be look like the content image.
2. Use a pre-trained ConvNet (E.g. VGG network) to calculate activation at each layer for both content image and generated image
3. The content cost is the just how similar of the content image activation and generated image activation.

To calculate style cost we need first to define the style of a image as correlation between activations across channels. 

<div align="center">
	<img src="resources/Screen Shot 2018-03-21 at 12.56.36 AM.png" width="70%">
	<img src="resources/Screen Shot 2018-03-21 at 12.55.49 AM.png" width="40%">
</div>

Then we will build a style matrix G for both style image G_s and generated image G_g, and the style cost will be:

<div align="center">
	<img src="resources/Screen Shot 2018-03-21 at 12.59.45 AM.png" width="70%">
</div>

To get a visually more pleasing result, we can sum over all the layers instead just one layer.