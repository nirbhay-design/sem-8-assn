library(tseries)
library(forecast)

data <- read.csv("CSVForDate.csv")
data_col <- data['Open']
data_col_diff <- diff(data_col[,1])

n_col <- ncol(data)
n_row <- nrow(data)

plot_fun <- function (p, x, funplot, plot_fun=FALSE, main='open') {
  png(p,width=547, height=337)
  if (plot_fun) {
    funplot(x,main=main,type='l')
  } else {
    funplot(x)
  }
  dev.off()
}

BPT <- function(x, lags) {
  for (i in 1:length(lags)) {
    cur_box_test <- Box.test(x, lag = lags[i], fitdf=0, type = c("Box-Pierce", "Ljung-Box"))
    print(cur_box_test)
  }
}

# analysing y

plot_fun("open/acf_open.png", data_col, acf)
plot_fun("open/pacf_open.png", data_col, pacf)
plot_fun("open/open.png", data_col[,1], plot, plot_fun=TRUE,main='open')

# test

adf.test(data_col[,1])
PP.test(data_col[,1])

# analysing first difference of y

plot_fun("open/acf_open_diff.png", data_col_diff, acf)
plot_fun("open/pacf_open_diff.png", data_col_diff, pacf)
plot_fun("open/open_diff.png", data_col_diff, plot, plot_fun=TRUE,main='open Diff')

# test first diff of y

adf.test(data_col_diff)
PP.test(data_col_diff)

# box pierce test for y

BPT(data_col, lags=c(1,2,3,4,10,15,17))

# arma modelling on y

arma22 <- arima(data_col,order=c(2,0,2))
arma22

arma20 <- arima(data_col,order=c(2,0,0))
arma20

arma02 <- arima(data_col,order=c(0,0,2))
arma02

# arch and garch modelling on y

garch22 <- garch(data_col,order=c(2,2))
garch22

garch02 <- garch(data_col, order=c(0,2))
garch02

# plotting residuals for each model

plot_fun(
  "open/arma22_open_res.png",
  arma22$residuals,
  plot,
  plot_fun=TRUE,
  main='arma22 residuals'
)

plot_fun(
  "open/arma20_open_res.png",
  arma20$residuals,
  plot,
  plot_fun=TRUE,
  main='arma20 residuals'
)

plot_fun(
  "open/arma02_open_res.png",
  arma02$residuals,
  plot,
  plot_fun=TRUE,
  main='arma02 residuals'
)

plot_fun(
  "open/garch22_open_res.png",
  garch22$residuals,
  plot,
  plot_fun=TRUE,
  main='garch22 residuals'
)

plot_fun(
  "open/garch02_open_res.png",
  garch02$residuals,
  plot,
  plot_fun=TRUE,
  main='garch02 residuals'
)


# forecasting using each arma model

n_forecast <- 1000
plot_fun(
  "open/forecast_open_arma22.png",
  forecast(arma22,n_forecast),
  plot,
  plot_fun=TRUE,
  main='arma22 forecast')

plot_fun(
  "open/forecast_open_arma20.png",
  forecast(arma20,n_forecast),
  plot,
  plot_fun=TRUE,
  main='arma20 forecast')

plot_fun(
  "open/forecast_open_arma02.png",
  forecast(arma02,n_forecast),
  plot,
  plot_fun=TRUE,
  main='arma02 forecast')

# box pierce test for arima and garch model residuals

BPT(arma22$residuals, lags=c(1,2,3,10,11))
BPT(arma20$residuals, lags=c(1,2,3,10,11))
BPT(arma02$residuals, lags=c(1,2,3,10,11))
BPT(garch22$residuals, lags=c(1,2,3,10,11))
BPT(garch02$residuals, lags=c(1,2,3,10,11))




