library(readr)
library(stringi)
library(keras)

# zdania <- read_tsv('slowianskie.csv', col_names =F, quote='', n_max = 8e6, col_types = 'ifc')
# laczenia <- read_tsv('links.csv', col_names = F)
# pl <- zdania[zdania$X2=='pol',]
# cz <- zdania[zdania$X2=='ces',]
# ru <- zdania[zdania$X2=='rus',]
# uk <- zdania[zdania$X2=='ukr',]

# pl.ru <- merge(pl, laczenia, by.x = "X1", by.y = "X1")
# pl.ru <- merge(pl.ru, ru, by.x = "X2.y", by.y = "X1")
# pl.ru <- pl.ru[,c(4,6)]

# ru.uk <- merge(ru, laczenia, by.x = "X1", by.y = "X1")
# ru.uk <- merge(ru.uk, uk, by.x = "X2.y", by.y = "X1")
# ru.uk <- ru.uk[,c(4,6)]
# ru.uk[,1] <- stri_trans_general(ru.uk[,1], "cyrillic-latin")
# ru.uk[,2] <- stri_trans_general(ru.uk[,2], "cyrillic-latin")

# pl.uk <- merge(pl, laczenia, by.x = "X1", by.y = "X1")
# pl.uk <- merge(pl.uk, uk, by.x = "X2.y", by.y = "X1")
# pl.uk <- pl.uk[,c(4,6)]
# pl.uk[,2] <- stri_trans_general(pl.uk[,2], "cyrillic-latin")
# write.table(pl.uk, "pl_uk.csv", quote = F, sep='\t', row.names = F, col.names = F, fileEncoding = "UTF-8")
# remove(laczenia, pl, uk, zdania)
wierszy <- 20000
maks_zn <- 25

dane1 <- read_tsv('rus.txt', quote='', n_max = wierszy, col_names = F)
dane1$X2 <- stri_trans_general(dane1$X2, "cyrillic-latin")
dane2 <- read_tsv('ukr.txt', quote='', n_max = wierszy, col_names = F)
dane2$X2 <- stri_trans_general(dane2$X2, "cyrillic-latin")
dane <- merge(dane1,dane2, by.x = 'X1', by.y = 'X1')
dane <- dane[,2:3]
dane <- dane[which(nchar(dane$X2.x)<=maks_zn),]
dane <- dane[which(nchar(dane$X2.y)<=maks_zn),]
maks_slow <- 6000

tokenizer_we <- text_tokenizer()
tokenizer_we %>% fit_text_tokenizer(unlist(dane[,1], use.names=F))
we <- tokenizer_we %>% texts_to_sequences(unlist(dane[,1], use.names=F))
we <- pad_sequences(we, padding = 'post')
sl_we <- unlist(tokenizer_we$index_word, use.names = F)
l_sl_we <- length(sl_we)

tokenizer_wy <- text_tokenizer(num_words = maks_slow)
tokenizer_wy %>% fit_text_tokenizer(unlist(dane[,2], use.names=F))
wy <- tokenizer_wy %>% texts_to_sequences(unlist(dane[,2], use.names=F))
wy <- pad_sequences(wy, padding = 'post')
sl_wy <- unlist(tokenizer_wy$index_word, use.names = F)
l_sl_wy <- min(length(sl_wy), maks_slow)

dim(wy) <- c(dim(wy),1)
id_wal <- sample.int(nrow(we), 100)
we_wal <- we[id_wal,,drop=F]
wy_wal <- wy[id_wal,,,drop=F]
we <- we[-id_wal,,drop=F]
wy <- wy[-id_wal,,,drop=F]

id_test <- sample.int(nrow(we), 0.1*nrow(we))
we_test <- we[id_test,,drop=F]
wy_test <- wy[id_test,,,drop=F]
we <- we[-id_test,,drop=F]
wy <- wy[-id_test,,,drop=F]

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = l_sl_we+1, output_dim = 256, input_length = dim(we)[2]) %>%
  bidirectional(layer_lstm(units=128)) %>%
  layer_repeat_vector(dim(wy)[2]) %>%
  bidirectional(layer_lstm(units=128, return_sequences = T)) %>%
  layer_dense(1024, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(l_sl_wy+1, activation = 'softmax') %>%
  compile(optimizer_adam(lr=0.01), 'sparse_categorical_crossentropy', list('acc'))

model %>% fit(we, wy, epochs=15, batch_size=256, validation_data = list(we_test, wy_test), view_metrics=T)

model %>% save_model_weights_hdf5('tlumacz2')

model %>% evaluate(we_wal, wy_wal)
pred <- model %>% predict(we_wal) %>% k_argmax() %>% k_get_value()
pred2 <- rep("", nrow(pred))
wz <- rep("", nrow(pred))
we2 <- rep("", nrow(pred))
for(i in 1:nrow(pred)){
  pred2[i] <- gsub('(poczatek)|(koniec)', '', paste0(sl_wy[pred[i,]], collapse = ' '))
  wz[i] <- gsub('(poczatek)|(koniec)', '', paste0(sl_wy[wy_wal[i,,1]], collapse = ' '))
  we2[i] <- paste0(sl_we[we_wal[i,]], collapse = ' ')
}
df <- data.frame(stri_trans_general(pred2, "latin-cyrillic"),
                 stri_trans_general(wz, "latin-cyrillic"),
                 stri_trans_general(we2, "latin-cyrillic"))
Sys.setlocale(locale="Russian")
write.table(df, 'wynik3.csv', quote=F, sep=';', row.names = F, col.names = F)

