# k-Means examples

This is the base repo for this [Medium post](https://medium.com/@paulo_sampaio/entendendo-k-means-agrupando-dados-e-tirando-camisas-e90ae3157c17), used for educational purposes - I needed to put together some k-Means examples to explain it to my students. There are 3 main files/examples:

## K-means image quantization

Colors are usually represented as a tuple of 3 8-bit integers (from 0 to 255), representing the intensities of red, green and blue. This means that there are around 16M possible colors (2<sup>8</sup> * 3) in modern image representations. But when I was a kid playing [Monkey Island](https://en.wikipedia.org/wiki/The_Secret_of_Monkey_Island) on my good old 286 computer, my EGA monitors used to have 16 colors. What if I wanted to "downgrade" current images back to 16 colors? Let's try to use k-Means for that.

## How it works

- I'm resizing the image so the largest side has 128 pixels (just so it runs faster)
- I'm converting the image from RGB to [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) (for me HSV makes more sense for clustering, since there is less correlation between dimensions)
- I'm reshaping the image to a pixel list (one row for each pixel, columns are the R, G and B values)
- Using k-Means to cluster into the number of colours you desire to see. 16 for instance.
- Get the label assigned to each one of the pixels and reshape it back to the original image shape, and color them using the cluster centroid.

That's it.
!["Lechuck"](https://paulo-blog-media.s3-sa-east-1.amazonaws.com/posts/2020-12-21-kmeans_examples/lechuck.jpg)Basically LeChuck

## How to run it

I built it into a [streamlit](https://www.streamlit.io/) app, you can easily run it with:

`streamlit run k_means_demo.py`

If you are into `docker`, I'm providing a `Dockerfile` and a `docker-compose.yml` file, so you can run it with:

`docker-compose up`

## Image segmentation notebook

Originally, this repo was just that notebook, about simple image segmentation with k-Means. This was part of what, back in 2014, I wrote in [my masters dissertation](https://upcommons.upc.edu/handle/2117/77312). At that time, I used k-Means combined with an [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) classifier to isolate segments and classify them into "product", "body" and "background". This notebook describes the process of image segmentation. It is very similar to quantizing, but using the least amount of clusters needed to retrieve the information.

## Wholesale customer notebook

This was the basic example, using [UCL wholesale dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers), just to familiarize students with what k-means is and what is its application in business.

## Closing thoughts

This is a repo for educational purposes, showcasing a couple of k-means interesting applications. Feel free to check it out and let me know what you think or if you have any other interesting use to share!
