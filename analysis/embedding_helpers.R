# EMBEDDING CALCULATIONS

embedding_norm <- function(x) sqrt(sum(x ^ 2))

embedding_cosine <- function(x, y) sum(x * y) / (embedding_norm(x) * embedding_norm(y))

cosine_gain <- function(request, output, target) {
  embedding_cosine(output, target) - embedding_cosine(request, target)
}
