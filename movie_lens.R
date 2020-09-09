library(tidyverse)


# Read data --------------------------------------------------------------------

#ml-1m dataset:
ratings <- readr::read_tsv("https://github.com/RuidongZ/Recurrent_Recommender_Networks/raw/master/data/ml-1m/ratings.dat", 
                           col_names = FALSE) %>%
  separate(col = 1, into = c("userID", "movieID", "rating", "timestamp")) %>%
  mutate_if(is.character, as.integer)
  

# Train/validation/test split --------------------------------------------------

test <- slice_max(ratings, order_by = timestamp, prop = 0.03)
train <- anti_join(ratings, test)
validation <- slice_min(test, order_by = timestamp, prop = 0.5)
test <- anti_join(test, validation)


# Define model -------------------------------------------------------------

source("RRN.R")
model <- rrn(num_users = max(ratings$userID), num_items = max(ratings$movieID))
                

# Train model -------------------------------------------------------------

# First define callbacks to stop model early when validation loss increases and to save best model
callback_list <- list(
  callback_early_stopping(patience = 2),
  callback_model_checkpoint(filepath = "model.h5", monitor = "val_loss", save_best_only = TRUE)
)

# Train model
history <- 
  model %>% 
  fit(
    x = list(user_input = as.array(train$userID), 
             item_input = as.array(train$movieID)),
    y = as.array(train$rating),
    epochs = 10,
    batch_size = 2048, 
    validation_data = list(list(user_input = as.array(validation$userID), 
                                item_input = as.array(validation$movieID)), 
                           as.array(validation$rating)),
    shuffle = TRUE, 
    callbacks = callback_list
  ) 

# Load best model:
model <- load_model_hdf5("my_model.h5")
                
                




