
var documents = [{
    "id": 0,
    "url": "http://localhost:4000/404.html",
    "title": "404",
    "body": "404 Page does not exist!Please use the search bar at the top or visit our homepage! "
    }, {
    "id": 1,
    "url": "http://localhost:4000/about",
    "title": "About me",
    "body": "Welcome to my site! My name is Aniket Sanap and I am a machine learning enthusiast. I am currently working as a machine learning engineer at AI Adventures and have been working there since June 2019. Before that I completed my Bachleor's of Engineering in Computer Engineering from Maharashtra Institute of Technology, Pune. I will be continuing my education in the form of a Master's degree from fall 2020. Get in touch!You can contact me on linkedin "
    }, {
    "id": 2,
    "url": "http://localhost:4000/categories",
    "title": "Categories",
    "body": ""
    }, {
    "id": 3,
    "url": "http://localhost:4000/",
    "title": "Home",
    "body": "      Featured:                                                                                                                                                                                                                 Content Based Image Retrieval (CBIR)                              :               Retrieving similar images based on the content of a query image:                                                                       28 Dec 2019                &lt;/span&gt;                                                                                                      Additional Projects:                                                                                                     Content Based Image Retrieval (CBIR)              :       Transferring the style of one image to another using neural networks:                               28 Dec 2019        &lt;/span&gt;                                    "
    }, {
    "id": 4,
    "url": "http://localhost:4000/robots.txt",
    "title": "",
    "body": "      Sitemap: {{ “sitemap. xml”   absolute_url }}   "
    }, {
    "id": 5,
    "url": "http://localhost:4000/Neural-Style-Transfer/",
    "title": "Content Based Image Retrieval (CBIR)",
    "body": "2019/12/28 - &lt;!DOCTYPE html&gt;   "
    }, {
    "id": 6,
    "url": "http://localhost:4000/Content-Based-Image-Retrieval/",
    "title": "Content Based Image Retrieval (CBIR)",
    "body": "2019/12/28 - Simple image classification was a challenge in computer vision not so long ago. All of this changed with the use of deep CNN architectures. Models like ResNet that use skip connections, leading to much deeper architectures have consistently shown impressive results on the ImageNet dataset. Due to the success of these models in other tasks through transfer learning, it is apparent that they are able to extract relevant information from an RGB image. In this post, we will attempt to use a ResNet which has been trained on ImageNet to extract relevant features from our dataset and use these features to find similar images. This is broadly known as “Content Based Image Retrieval” where similar images are found based on semantic similarity. To replicate these results you will need PyTorch, faiss, NumPy and matplotlib. If you just want the code, I have it on my github or nbviewer. For this project, we will use this Jewellery dataset. This dataset contains four classes:  Bracelets (309 images).  Earrings (472 images).  Necklaces (301 images).  Rings (189 images). The images have the jewellery item in focus with a white background. This is a very clean dataset and should give us good results. Downloading the dataset: 1234wget --no-check-certificate 'https://docs. google. com/uc?export=download&amp;id=0B4KI-B-t3wTjbElMTS1DVldQUnc' -O Jewellery. tar. gztar xvf Jewellery. tar. gzrm Jewellery/*. gzrm Jewellery/*. zipThe above lines of code will download and extract the dataset using the terminal. They can be run in a jupyter notebook as well by inserting a ‘!’ before each line. After running the above code, you should have a directory called Jewellery which contains 4 different subdirectories with the names of each of the 4 different classes. Sound familiar? This is because this is exactly the format required by the torchvision. datasets. ImageFolder class! Unfortunately, as of the writing of this blog, this class does not return the name of the file. Building a custom ImageFolder class: We can make one very small modification to the builtin ImageFolder class so that it also returns the filenames. We require the file names to have a mapping of images with their extracted features. 123456789101112131415class ImageFolderWithPaths(datasets. ImageFolder):     Custom dataset that includes image file paths. Extends  torchvision. datasets. ImageFolder  Source: https://gist. github. com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d         # override the __getitem__ method. this is the method that dataloader calls  def __getitem__(self, index):    # this is what ImageFolder normally returns     original_tuple = super(ImageFolderWithPaths, self). __getitem__(index)    # the image file path    path = self. imgs[index][0]    # make a new tuple that includes original and the path    tuple_with_path = (original_tuple + (path,))    return tuple_with_pathAs you can see from the above code, just by adding a couple of lines to the __getitem__ method, we are able to return the file names along with the image tensor and a label if necessary. Preprocessing the input data: We do not require a lot of preprocessing for this sample dataset. Here we will just resize the input images to (224 x 224) as that is the input size required by the ResNet. This can be achieved using a simple torchvision. transforms. Resize(). Hence this is what our preprocessing looks like: 1234567transforms_ = transforms. Compose([  transforms. Resize(size=[224, 224], interpolation=2),  transforms. ToTensor()])dataset = ImageFolderWithPaths('Jewellery', transforms_) # our custom datasetdataloader = torch. utils. data. DataLoader(dataset, batch_size=1)Downloading the model: We will be using the pretrained ResNet50 from torchvision. models. You can try using the same logic on multiple different CNN architectures but we will be using ResNet50 for this blog. 12DEVICE = 'cuda' if torch. cuda. is_available() else 'cpu'model = models. resnet50(pretrained=True)ResNet is by default used for classification. We don’t want the output from the output layer of the ResNet. We will consider our feature vector to be the output of the last pooling layer. To extract the output from this pooling layer, we will use a small function: 1234567def pooling_output(x):  global model  for layer_name, layer in model. _modules. items():    x = layer(x)    if layer_name == 'avgpool':      break  return xHere avgpool is the name of the last pooling layer in the structure of our model. Creating feature vectors: We now have everything we require to create our feature vectors. This is a very straightforward process. Make sure you put the model in eval() mode before running this! 1234567891011# iterate over dataimage_paths = []descriptors = []model. to(DEVICE)with torch. no_grad():  model. eval()  for inputs, labels, paths in dataloader:    result = pooling_output(inputs. to(DEVICE))    descriptors. append(result. cpu(). view(1, -1). numpy())    image_paths. append(paths)    torch. cuda. empty_cache()Once this code finishes execution, congratulations! You have now built feature vectors from your dataset. But how do you find similar images from these feature vectors? This is where faiss comes in. The description of faiss from its github is “A library for efficient similarity search and clustering of dense vectors”. This is a library created by Facebook which is super fast at similarity search, which is exactly what we want. Installing faiss: 1234wget https://anaconda. org/pytorch/faiss-gpu/1. 2. 1/download/linux-64/faiss-gpu-1. 2. 1-py36_cuda9. 0. 176_1. tar. bz2tar xvjf faiss-gpu-1. 2. 1-py36_cuda9. 0. 176_1. tar. bz2cp -r lib/python3. 6/site-packages/* /usr/local/lib/python3. 6/dist-packages/pip install mklYou may want to replace my version with the latest one. But I cannot promise that it will work the same, so in case of any errors, try installing the same version of faiss that I have. Creating a faiss index: The way that we will use faiss is that first we will create a faiss index using our precalculated feature vectors. Then at runtime we will get another image. We will then run this image through our model and calculate its feature vector as well. We will then query faiss with the new feature vector to find similar vectors. It should be clearer with code. 1234567import numpy as npimport faissindex = faiss. IndexFlatL2(2048)descriptors = np. vstack(descriptors)index. add(descriptors)Calculating the feature vector of a query image and searching using faiss: 12345678query_image = 'Jewellery/bracelet/bracelet_048. jpg'img = Image. open(query_image)input_tensor = transforms_(img)input_tensor = input_tensor. view(1, *input_tensor. shape)with torch. no_grad():  query_descriptors = pooling_output(input_tensor. to(DEVICE)). cpu(). numpy()  distance, indices = index. search(query_descriptors. reshape(1, 2048), 9)Using the above piece of code, I got the following results:    Query image:     Top 9 results:  The results are not that bad! The first image is just the query image as naturally it will have the most similar vector. The rest of the images are what I would say pretty similar to the query image. This is especially apparent because of the circular piece of jewellery at the center of the bracelet. But I would say for a model not trained at all on this specific dataset, the results are acceptable. You can try training the model on the actual dataset, augmenting the images, adding a bit of noise to make the model a bit more general or any other technique you want to try and improve the performance of the model. The complete code for this project is available in the form of a jupyter notebook on my github or on nbviewer. You can leave any questions, comments or concerns in the comment section below. I hope this post was useful :) "
    }];

var idx = lunr(function () {
    this.ref('id')
    this.field('title')
    this.field('body')

    documents.forEach(function (doc) {
        this.add(doc)
    }, this)
});
function lunr_search(term) {
    document.getElementById('lunrsearchresults').innerHTML = '<ul></ul>';
    if(term) {
        document.getElementById('lunrsearchresults').innerHTML = "<p>Search results for '" + term + "'</p>" + document.getElementById('lunrsearchresults').innerHTML;
        //put results on the screen.
        var results = idx.search(term);
        if(results.length>0){
            //console.log(idx.search(term));
            //if results
            for (var i = 0; i < results.length; i++) {
                // more statements
                var ref = results[i]['ref'];
                var url = documents[ref]['url'];
                var title = documents[ref]['title'];
                var body = documents[ref]['body'].substring(0,160)+'...';
                document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML + "<li class='lunrsearchresult'><a href='" + url + "'><span class='title'>" + title + "</span><br /><span class='body'>"+ body +"</span><br /><span class='url'>"+ url +"</span></a></li>";
            }
        } else {
            document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = "<li class='lunrsearchresult'>No results found...</li>";
        }
    }
    return false;
}

function lunr_search(term) {
    $('#lunrsearchresults').show( 400 );
    $( "body" ).addClass( "modal-open" );
    
    document.getElementById('lunrsearchresults').innerHTML = '<div id="resultsmodal" class="modal fade show d-block"  tabindex="-1" role="dialog" aria-labelledby="resultsmodal"> <div class="modal-dialog shadow-lg" role="document"> <div class="modal-content"> <div class="modal-header" id="modtit"> <button type="button" class="close" id="btnx" data-dismiss="modal" aria-label="Close"> &times; </button> </div> <div class="modal-body"> <ul class="mb-0"> </ul>    </div> <div class="modal-footer"><button id="btnx" type="button" class="btn btn-danger btn-sm" data-dismiss="modal">Close</button></div></div> </div></div>';
    if(term) {
        document.getElementById('modtit').innerHTML = "<h5 class='modal-title'>Search results for '" + term + "'</h5>" + document.getElementById('modtit').innerHTML;
        //put results on the screen.
        var results = idx.search(term);
        if(results.length>0){
            //console.log(idx.search(term));
            //if results
            for (var i = 0; i < results.length; i++) {
                // more statements
                var ref = results[i]['ref'];
                var url = documents[ref]['url'];
                var title = documents[ref]['title'];
                var body = documents[ref]['body'].substring(0,160)+'...';
                document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML + "<li class='lunrsearchresult'><a href='" + url + "'><span class='title'>" + title + "</span><br /><small><span class='body'>"+ body +"</span><br /><span class='url'>"+ url +"</span></small></a></li>";
            }
        } else {
            document.querySelectorAll('#lunrsearchresults ul')[0].innerHTML = "<li class='lunrsearchresult'>Sorry, no results found. Close & try a different search!</li>";
        }
    }
    return false;
}
    
$(function() {
    $("#lunrsearchresults").on('click', '#btnx', function () {
        $('#lunrsearchresults').hide( 5 );
        $( "body" ).removeClass( "modal-open" );
    });
});