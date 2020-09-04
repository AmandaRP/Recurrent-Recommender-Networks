library(keras)

# Define network inputs -----------------------------------------------------------

rrn <- function(data, user_embedding_dim, item_embedding_dim, hidden_size, out_size){
  
}

#pass in user/item indexes (not one-hot vectors) b/c this is required by embedding layer
user_input <- layer_input(shape=1, name = "user_input") 
item_input <- layer_input(shape=1, name = "item_input")
#rating_input <- 

user_vector <- user_input %>%  
  layer_embedding(input_dim = num_users, # "dictionary" size
                  output_dim = user_embedding_dim,
                  #embeddings_initializer = initializer_random_normal(0, sigma), # Use N(0,sigma) initialization  
                  #embeddings_regularizer = regularizer_l2(lambda), 
                  input_length = 1,  # the length of the sequence that is being fed in (one integer)
                  name = "user_embedding") %>%
  layer_lstm(units = hidden_size, name = "user_lstm") %>%
  layer_dense(units = out_size)
#TODO: Flatten?

item_vector <- item_input %>%  
  layer_embedding(input_dim = num_items, # "dictionary" size
                  output_dim = item_embedding_dim,
                  #embeddings_initializer = initializer_random_normal(0, sigma), # Use N(0,sigma) initialization  
                  #embeddings_regularizer = regularizer_l2(lambda), 
                  input_length = 1,  # the length of the sequence that is being fed in (one integer)
                  name = "item_embedding") %>%
  layer_lstm(units = hidden_size, name = "item_lstm") %>%
  layer_dense(units = out_size)


prediction <- layer_dot(c(user_vector, item_vector), name = "prediction")

model <- keras_model(list(user_input, item_input), prediction)

# Compile model ---------------------------------------------------

model %>% compile(
  optimizer = "rmsprop",
  loss = "mse", 
  metrics = c("mse")
)

summary(model)

  