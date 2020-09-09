library(keras)

# Define network ---------------------------------------------------------------

rrn <- function(num_users, num_items, user_embedding_dim = 40, item_embedding_dim = 40, hidden_size = 40, out_size = 20){
  
  #pass in user/item indexes
  user_input <- layer_input(shape=1, name = "user_input") 
  item_input <- layer_input(shape=1, name = "item_input")
  
  user_vector <- user_input %>%  
    layer_embedding(input_dim = num_users, # "dictionary" size
                    output_dim = user_embedding_dim,
                    #embeddings_regularizer = regularizer_l2(lambda), 
                    input_length = 1,  # the length of the sequence that is being fed in (one integer)
                    name = "user_embedding") %>%
    layer_lstm(units = hidden_size, name = "user_lstm") %>%
    layer_dense(units = out_size)
  
  item_vector <- item_input %>%  
    layer_embedding(input_dim = num_items, # "dictionary" size
                    output_dim = item_embedding_dim,
                    #embeddings_regularizer = regularizer_l2(lambda), 
                    input_length = 1,  
                    name = "item_embedding") %>%
    layer_lstm(units = hidden_size, name = "item_lstm") %>%
    layer_dense(units = out_size)
  
  
  prediction <- layer_dot(c(user_vector, item_vector), axes = 1, name = "prediction") #TODO: Correct axes?
  
  model <- keras_model(list(user_input, item_input), prediction)
  
  # Compile model ---------------------------------------------------
  
  model %>% compile(
    optimizer = "adam",
    loss = "mse", 
    metrics = c("rmse")
  )
  
  summary(model)
  
}


  