Deep Learning Based Computer Vision - Waste Sorting (with Source Code)
1. Introduction to the experiment
1.1 Experimental background
From July 7 this year, Shanghai will officially implement the Shanghai Municipal Household Waste Management Regulations. Garbage sorting, seemingly a trivial "trivial matter", is actually related to the improvement of the living environment of more than 1.13 billion people, and should be vigorously advocated. The garbage identification and classification data includes glass, cardboard, metal, paper, plastic, and general garbage, a total of 6 categories. Due to the wide variety of domestic waste, the lack of unified standards for specific classification, most people will "choose difficult" in actual operation, based on deep learning technology to establish an accurate classification model, the use of technical means to improve the living environment.

1.2 Experimental Requirements
a) Build a deep neural network model and tune it to the best possible state. b) Graph deep neural network models, plot and analyze learning curves. c) Evaluate the model with indicators such as accuracy.

1.3 Experimental Environment
You can use the Python-based OpenCV library for image correlation processing, the Numpy library for related numerical operations, and frameworks such as Keras to build deep learning models.

1.4 References
OpenCV：https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html Numpy：https://www.numpy.org/ Keras: https://keras.io/

2. Experimental content
2.1 Introducing datasets
The dataset contains 2507 images of household waste. The creators of the dataset divide garbage into 6 categories, which are:

serial number	Chinese name	English name	Dataset size
1	glass	glass	497 total images
2	paper	paper	590 total images
3	cardboard	cardboard	400 total images
4	plastics	plastic	479 total images
5	metal	metal	407 total images
6	General garbage	trash	134 total images
The items are all taken on a whiteboard under daylight/indoor light source, and the compressed size is 512*384

Image preview:

 ![68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303139313130333138343630333832342e706e673f782d6f73732d70726f636573733d696d6167652f77617465726d61726b2c747970655f5a6d46755a33706f5a57356e6147567064476b2c](https://user-images.githubusercontent.com/94170592/231307311-62739c82-a0a3-41ef-812a-31640dae8de4.png)


 some of the predicted result show:
 
![dd](https://user-images.githubusercontent.com/94170592/231307202-07692340-34c5-4d3f-88c9-3de2790d99bb.png)

