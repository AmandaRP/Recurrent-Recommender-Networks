library(tidyverse)

#ml-1m dataset:
ratings <- readr::read_tsv("https://github.com/RuidongZ/Recurrent_Recommender_Networks/raw/master/data/ml-1m/ratings.dat", col_names = FALSE) %>%
  separate(col = "X1", into = c("userID", "movieID", "rating", "timestamp")) %>%
  mutate_if(is.character, as.integer)
  