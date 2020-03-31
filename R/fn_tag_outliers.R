tag_outliers <- function(x, na.rm = TRUE, ...) {
    x %in% boxplot.stats(x)$out
}
