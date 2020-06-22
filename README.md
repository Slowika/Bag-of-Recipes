# Bag-of-Recipes

For this data mining project we developed a recommendation system to suggest ingredients that can be added to a partial recipe, for which we used a dataset of recipes from yummly.co.uk. We explored the effect of different similarity measures and dimensionality reduction on the quality of the recommendations. For the evaluation, we removed one ingredient from each recipe and used our recommendation system to recover it. Our best method placed this hidden ingredient in the top10 recommendations more than 40% of the times.


# Citation 

If you find this project useful, please cite:

```
@article{DBLP:journals/corr/abs-1907-12380,
  author    = {Paula Ferm{\'{\i}}n Cueto and
               Meeke Roet and
               Agnieszka Slowik},
  title     = {Completing partial recipes using item-based collaborative filtering
               to recommend ingredients},
  journal   = {CoRR},
  volume    = {abs/1907.12380},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.12380},
  archivePrefix = {arXiv},
  eprint    = {1907.12380},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1907-12380.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


# License

This implementation is licensed under the MIT license. The text of the license can be found [here](https://github.com/Slowika/Bag-of-Recipes/blob/master/LICENSE).




